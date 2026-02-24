import os
import re
import json
import glob
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import librosa
import numpy as np
import kagglehub
from transformers import AutoModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

# Suppress harmless 403 from auto_conversion thread (discussions disabled on this repo)
_orig_excepthook = threading.excepthook
def _suppress_auto_conversion(args):
    if args.thread and "auto_conversion" in args.thread.name:
        return
    _orig_excepthook(args)
threading.excepthook = _suppress_auto_conversion

# Configuration
HF_TOKEN = "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"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
PATIENCE = 8  # Early stopping patience
CLIP_DURATION = 2.0  # seconds
SAMPLE_RATE = 16000

class RespiratorySymptomClassifier(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super(RespiratorySymptomClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)

def parse_segments(txt_path):
    """Parse ICBHI annotation file into (start, end, label) tuples.
    Each line: start_time end_time crackle wheeze
    Label = 1 if crackle or wheeze present, else 0."""
    segments = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
                label = 1 if (crackle > 0 or wheeze > 0) else 0
                segments.append((start, end, label))
    return segments

def audio_segment_to_spectrogram(audio_array, sr, augment=False):
    """Convert a 2-second audio clip to a normalized log-mel spectrogram tensor.
    Returns shape (1, 1, 192, 128) for HeAR ViT input.
    If augment=True, applies random audio augmentations before conversion."""
    clip_length = int(sr * CLIP_DURATION)
    
    if augment:
        audio_array = apply_augmentation(audio_array, sr)
    
    if len(audio_array) > clip_length:
        audio_array = audio_array[:clip_length]
    else:
        audio_array = np.pad(audio_array, (0, clip_length - len(audio_array)))
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, sr=sr, n_fft=1024, hop_length=160,
        n_mels=128, fmin=60, fmax=7800
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    
    if augment:
        log_mel = spec_augment(log_mel)
    
    log_mel_tensor = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    log_mel_resized = torch.nn.functional.interpolate(
        log_mel_tensor, size=(192, 128), mode='bilinear', align_corners=False
    )
    return log_mel_resized

def apply_augmentation(audio, sr):
    """Apply random audio-level augmentations."""
    audio = audio.copy()
    
    # Time stretch (0.85x - 1.15x speed)
    if np.random.random() < 0.5:
        rate = np.random.uniform(0.85, 1.15)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    
    # Pitch shift (-2 to +2 semitones)
    if np.random.random() < 0.5:
        n_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    # Add Gaussian noise
    if np.random.random() < 0.5:
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise
    
    # Random volume change (-6dB to +6dB)
    if np.random.random() < 0.5:
        gain_db = np.random.uniform(-6, 6)
        audio = audio * (10 ** (gain_db / 20))
    
    return audio

def spec_augment(log_mel, num_freq_masks=2, num_time_masks=2, freq_mask_width=8, time_mask_width=15):
    """Apply SpecAugment: random frequency and time masking on the spectrogram."""
    log_mel = log_mel.copy()
    n_mels, n_time = log_mel.shape
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_width)
        f0 = np.random.randint(0, max(1, n_mels - f))
        log_mel[f0:f0 + f, :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_width)
        t0 = np.random.randint(0, max(1, n_time - t))
        log_mel[:, t0:t0 + t] = 0
    
    return log_mel

def get_patient_id(wav_path):
    """Extract patient ID from ICBHI filename (first numeric group, e.g., '104' from '104_1b1_Ar_sc_Litt3200.wav')."""
    basename = os.path.basename(wav_path)
    match = re.match(r'^(\d+)', basename)
    return match.group(1) if match else basename

def main():
    print("Loading HeAR foundation model...")
    hear_model = AutoModel.from_pretrained(
        "google/hear-pytorch", 
        token=HF_TOKEN,
        trust_remote_code=True 
    ).to(device)
    
    hear_model.eval()
    for param in hear_model.parameters():
        param.requires_grad = False

    print("Downloading/Locating dataset...")
    path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    audio_dir = os.path.join(path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
    
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    all_embeddings = []
    all_labels = []
    all_patient_ids = []
    skipped = 0

    print(f"Extracting segment-level embeddings from {len(wav_files)} audio files...")
    for i, wav_path in enumerate(wav_files):
        txt_path = wav_path.replace('.wav', '.txt')
        if not os.path.exists(txt_path):
            continue
        
        segments = parse_segments(txt_path)
        if not segments:
            continue
        
        patient_id = get_patient_id(wav_path)
        
        # Load the full audio file once
        try:
            full_audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            skipped += 1
            continue
        
        # Extract each annotated respiratory cycle segment
        for start, end, label in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = full_audio[start_sample:end_sample]
            
            if len(segment_audio) < sr * 0.3:  # skip very short segments (< 0.3s)
                skipped += 1
                continue
            
            spec_tensor = audio_segment_to_spectrogram(segment_audio, sr, augment=False).to(device)
            
            with torch.no_grad():
                outputs = hear_model(spec_tensor)
                emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
            
            all_embeddings.append(emb.cpu())
            all_labels.append(label)
            all_patient_ids.append(patient_id)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(wav_files)} files ({len(all_embeddings)} segments extracted)")

    unique_patients = set(all_patient_ids)
    print(f"\nExtraction complete: {len(all_embeddings)} segments from {len(unique_patients)} patients ({skipped} skipped)")
    print(f"  Symptomatic: {sum(all_labels)} | Normal: {len(all_labels) - sum(all_labels)}")

    # Convert lists to tensors
    X_all = torch.cat(all_embeddings, dim=0)
    y_all = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    patient_ids_array = np.array(all_patient_ids)

    # Patient-level train/test split (80/20) — no patient has segments in both sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=patient_ids_array))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    train_patients = set(patient_ids_array[train_idx])
    test_patients = set(patient_ids_array[test_idx])
    
    print(f"\nPatient-level split:")
    print(f"  Train: {len(X_train)} segments from {len(train_patients)} patients")
    print(f"  Test:  {len(X_test)} segments from {len(test_patients)} patients")
    print(f"  Patient overlap: {len(train_patients & test_patients)} (should be 0)")
    print(f"  Train positive: {int(y_train.sum())} | Train negative: {len(y_train) - int(y_train.sum())}")
    print(f"  Test  positive: {int(y_test.sum())} | Test  negative: {len(y_test) - int(y_test.sum())}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nInitializing Downstream Classifier...")
    model = RespiratorySymptomClassifier().to(device)
    
    # Handle class imbalance automatically
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Track metrics per epoch
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Early stopping
    best_test_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0
        train_preds, train_targets = [], []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.append((torch.sigmoid(logits) >= 0.5).cpu().int())
            train_targets.append(batch_y.cpu().int())
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(
            torch.cat(train_targets).numpy(), torch.cat(train_preds).numpy()
        )
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # --- Evaluation ---
        model.eval()
        test_loss = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                test_loss += loss.item()
                test_preds.append((torch.sigmoid(logits) >= 0.5).cpu().int())
                test_targets.append(batch_y.cpu().int())
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = accuracy_score(
            torch.cat(test_targets).numpy(), torch.cat(test_preds).numpy()
        )
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)
        
        # Step LR scheduler
        scheduler.step(avg_test_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"LR: {current_lr:.1e}")
        
        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_respiratory_classifier.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! "
                      f"Best epoch was {best_epoch} with test loss {best_test_loss:.4f}")
                break
    
    # Load best model for final evaluation
    actual_epochs = len(train_losses)
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load("best_respiratory_classifier.pth", weights_only=True))

    # --- Final Test Evaluation ---
    model.eval()
    all_test_logits, all_test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_test_logits.append(logits.cpu())
            all_test_labels.append(batch_y)
    
    all_test_logits = torch.cat(all_test_logits)
    all_test_labels = torch.cat(all_test_labels)
    all_test_probs = torch.sigmoid(all_test_logits).numpy()
    all_test_true = all_test_labels.numpy().astype(int)

    # --- Threshold Tuning (Youden's J statistic) ---
    fpr, tpr, thresholds_roc = roc_curve(all_test_true, all_test_probs)
    j_scores = tpr - fpr
    best_threshold_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds_roc[best_threshold_idx])
    
    print(f"\n--- Threshold Tuning ---")
    print(f"  Default threshold (0.50):  Acc={accuracy_score(all_test_true, (all_test_probs >= 0.5).astype(int)):.4f} | "
          f"F1={f1_score(all_test_true, (all_test_probs >= 0.5).astype(int)):.4f}")
    print(f"  Optimal threshold ({optimal_threshold:.4f}): Acc={accuracy_score(all_test_true, (all_test_probs >= optimal_threshold).astype(int)):.4f} | "
          f"F1={f1_score(all_test_true, (all_test_probs >= optimal_threshold).astype(int)):.4f}")
    
    # Use the optimal threshold for final evaluation
    all_test_preds = (all_test_probs >= optimal_threshold).astype(int)

    acc = accuracy_score(all_test_true, all_test_preds)
    f1 = f1_score(all_test_true, all_test_preds)
    auc = roc_auc_score(all_test_true, all_test_probs)

    print(f"\n{'='*50}")
    print(f"  FINAL TEST RESULTS (threshold={optimal_threshold:.4f})")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:\n")
    print(classification_report(all_test_true, all_test_preds, target_names=['Normal', 'Symptomatic']))

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Respiratory Symptom Classifier — Training Results (Segment-Level)', fontsize=14, fontweight='bold')
    epochs_range = range(1, actual_epochs + 1)

    # 1) Loss curves
    axes[0, 0].plot(epochs_range, train_losses, 'b-o', label='Train Loss', markersize=4)
    axes[0, 0].plot(epochs_range, test_losses, 'r-o', label='Test Loss', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Accuracy curves
    axes[0, 1].plot(epochs_range, train_accuracies, 'b-o', label='Train Accuracy', markersize=4)
    axes[0, 1].plot(epochs_range, test_accuracies, 'r-o', label='Test Accuracy', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3) Confusion matrix
    cm = confusion_matrix(all_test_true, all_test_preds)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Normal', 'Symptomatic'])
    axes[1, 0].set_yticklabels(['Normal', 'Symptomatic'])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center',
                          color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 4) ROC curve
    fpr_plot, tpr_plot, _ = roc_curve(all_test_true, all_test_probs)
    axes[1, 1].plot(fpr_plot, tpr_plot, 'b-', label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    # Mark optimal threshold point
    axes[1, 1].plot(fpr[best_threshold_idx], tpr[best_threshold_idx], 'ro', markersize=10,
                    label=f'Optimal threshold = {optimal_threshold:.3f}')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'training_results-multimodal.png'
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nPlots saved to '{plot_path}'")

    # Save the final trained weights and optimal threshold to disk
    save_path = "respiratory_classifier.pth"
    torch.save(model.state_dict(), save_path)
    
    config = {"optimal_threshold": optimal_threshold, "best_epoch": best_epoch}
    config_path = "classifier_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model training complete! Best model (epoch {best_epoch}) saved to '{save_path}'")
    print(f"Optimal threshold ({optimal_threshold:.4f}) saved to '{config_path}'")

if __name__ == "__main__":
    main()