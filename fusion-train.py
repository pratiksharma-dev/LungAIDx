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
import pandas as pd
import joblib
import kagglehub
from transformers import AutoModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

# Suppress harmless 403 from auto_conversion thread
_orig_excepthook = threading.excepthook
def _suppress_auto_conversion(args):
    if args.thread and "auto_conversion" in args.thread.name:
        return
    _orig_excepthook(args)
threading.excepthook = _suppress_auto_conversion

# ==========================================
# Configuration
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
PATIENCE = 10
CLIP_DURATION = 2.0
SAMPLE_RATE = 16000

# ==========================================
# 1. Model Architectures
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))

class TabularFeatureExtractor(nn.Module):
    """Same as LungRiskNet but stops at the 128-dim layer (removes final Linear(128,1))."""
    def __init__(self, input_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(512, dropout=0.3),
            ResidualBlock(512, dropout=0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            ResidualBlock(256, dropout=0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.features(x)

class AudioFeatureExtractor(nn.Module):
    """Same as RespiratorySymptomClassifier but stops at 64-dim layer."""
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.GELU(), nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.features(x)

class MultimodalFusionNet(nn.Module):
    def __init__(self, tab_input_dim, tab_weights_path, aud_weights_path):
        super().__init__()
        self.tab_base = TabularFeatureExtractor(tab_input_dim)
        self.aud_base = AudioFeatureExtractor()

        # Load pre-trained weights (ignoring removed final layers)
        tab_state = torch.load(tab_weights_path, map_location='cpu', weights_only=False)['model_state_dict']
        aud_state = torch.load(aud_weights_path, map_location='cpu', weights_only=True)

        # LungRiskNet final layer is net.15 (Linear(128,1)); filter it out
        tab_state = {k.replace('net.', 'features.'): v for k, v in tab_state.items() if 'net.15' not in k}
        # RespiratorySymptomClassifier final layer is classifier.12; filter it out
        aud_state = {k.replace('classifier.', 'features.'): v for k, v in aud_state.items() if 'classifier.12' not in k}

        self.tab_base.load_state_dict(tab_state, strict=False)
        self.aud_base.load_state_dict(aud_state, strict=False)

        # Freeze base models — only train fusion head
        for param in self.tab_base.parameters():
            param.requires_grad = False
        for param in self.aud_base.parameters():
            param.requires_grad = False

        # Fusion Head: 128 (tabular) + 64 (audio) = 192
        self.fusion_head = nn.Sequential(
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_tab, x_aud):
        feat_tab = self.tab_base(x_tab)
        feat_aud = self.aud_base(x_aud)
        combined = torch.cat((feat_tab, feat_aud), dim=1)
        return self.fusion_head(combined)

# ==========================================
# 2. Audio Preprocessing
# ==========================================
def parse_segments(txt_path):
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

def audio_segment_to_spectrogram(audio_array, sr):
    clip_length = int(sr * CLIP_DURATION)
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

    t = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.interpolate(t, size=(192, 128), mode='bilinear', align_corners=False)

def get_patient_id(wav_path):
    basename = os.path.basename(wav_path)
    match = re.match(r'^(\d+)', basename)
    return int(match.group(1)) if match else 0

# ==========================================
# 3. Synthetic Tabular Profile Generator
# ==========================================
# Maps ICBHI patient diagnosis to realistic tabular risk features
DIAGNOSIS_PROFILES = {
    'COPD':           {'age': (55, 80), 'pack_years': (20, 80), 'copd_diagnosis': 1, 'risk': 'high'},
    'Asthma':         {'age': (20, 60), 'pack_years': (0, 15),  'copd_diagnosis': 0, 'risk': 'moderate'},
    'Bronchiectasis': {'age': (40, 75), 'pack_years': (5, 40),  'copd_diagnosis': 0, 'risk': 'moderate'},
    'Pneumonia':      {'age': (30, 80), 'pack_years': (0, 30),  'copd_diagnosis': 0, 'risk': 'moderate'},
    'Bronchiolitis':  {'age': (1, 10),  'pack_years': (0, 0),   'copd_diagnosis': 0, 'risk': 'low'},
    'URTI':           {'age': (10, 50), 'pack_years': (0, 10),  'copd_diagnosis': 0, 'risk': 'low'},
    'LRTI':           {'age': (20, 60), 'pack_years': (0, 15),  'copd_diagnosis': 0, 'risk': 'low'},
    'Healthy':        {'age': (18, 70), 'pack_years': (0, 5),   'copd_diagnosis': 0, 'risk': 'low'},
}

def generate_tabular_profile(diagnosis, rng=None):
    """Generate a single synthetic tabular row matching the ICBHI patient's diagnosis."""
    if rng is None:
        rng = np.random.default_rng()

    profile = DIAGNOSIS_PROFILES.get(diagnosis, DIAGNOSIS_PROFILES['Healthy'])
    age_lo, age_hi = profile['age']
    py_lo, py_hi = profile['pack_years']

    age = rng.integers(age_lo, age_hi + 1)
    pack_years = rng.uniform(py_lo, py_hi)
    copd = profile['copd_diagnosis']

    # Correlate other risk factors with diagnosis severity
    is_high_risk = profile['risk'] == 'high'

    gender = rng.choice([0.0, 1.0])
    radon = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2] if not is_high_risk else [0.2, 0.3, 0.5])
    asbestos = rng.choice([0, 1], p=[0.8, 0.2] if not is_high_risk else [0.4, 0.6])
    secondhand = rng.choice([0, 1], p=[0.6, 0.4] if not is_high_risk else [0.3, 0.7])
    alcohol = rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2] if not is_high_risk else [0.2, 0.3, 0.5])
    family = rng.choice([0, 1], p=[0.7, 0.3] if not is_high_risk else [0.4, 0.6])

    return {
        'age': float(age),
        'gender': gender,
        'pack_years': pack_years,
        'radon_exposure': float(radon),
        'asbestos_exposure': float(asbestos),
        'secondhand_smoke_exposure': float(secondhand),
        'copd_diagnosis': float(copd),
        'alcohol_consumption': float(alcohol),
        'family_history': float(family),
    }

def compute_fusion_label(audio_label, tabular_profile):
    """Compute fusion label: 1 if EITHER modality indicates risk.
    Audio: symptomatic breathing. Tabular: high-risk demographic profile."""
    tab_risk = (tabular_profile['pack_years'] > 20 and
                tabular_profile['age'] > 45 and
                (tabular_profile['copd_diagnosis'] == 1 or
                 tabular_profile['family_history'] == 1))
    return 1 if (audio_label == 1 or tab_risk) else 0


# ==========================================
# 4. Main Training Pipeline
# ==========================================
def main():
    print(f"Fusion Training on {device}")
    print("=" * 60)

    # --- Load patient diagnosis metadata ---
    print("\nLoading datasets...")
    audio_path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    audio_dir = os.path.join(audio_path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
    diag_csv = os.path.join(audio_path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "patient_diagnosis.csv")

    patient_diag = {}
    diag_df = pd.read_csv(diag_csv, header=None, names=['patient_id', 'diagnosis'])
    for _, row in diag_df.iterrows():
        patient_diag[int(row['patient_id'])] = row['diagnosis'].strip()

    # --- Load preprocessing pipeline ---
    preprocessor = joblib.load('preprocessor.pkl')
    poly = joblib.load('poly_features.pkl')

    # --- Load HeAR model ---
    print("Loading HeAR foundation model...")
    hear_model = AutoModel.from_pretrained(
        "google/hear-pytorch", token=HF_TOKEN, trust_remote_code=True
    ).to(device)
    hear_model.eval()
    for param in hear_model.parameters():
        param.requires_grad = False

    # --- Extract paired (tabular, audio, label) for each segment ---
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    rng = np.random.default_rng(42)

    all_tab_features = []
    all_aud_embeddings = []
    all_labels = []
    all_patient_ids = []
    skipped = 0

    print(f"Extracting paired embeddings from {len(wav_files)} audio files...")
    for i, wav_path in enumerate(wav_files):
        txt_path = wav_path.replace('.wav', '.txt')
        if not os.path.exists(txt_path):
            continue

        segments = parse_segments(txt_path)
        if not segments:
            continue

        pid = get_patient_id(wav_path)
        diagnosis = patient_diag.get(pid, 'Healthy')

        try:
            full_audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            skipped += 1
            continue

        for start, end, audio_label in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = full_audio[start_sample:end_sample]

            if len(segment_audio) < sr * 0.3:
                skipped += 1
                continue

            # Generate paired tabular profile
            tab_profile = generate_tabular_profile(diagnosis, rng)
            fusion_label = compute_fusion_label(audio_label, tab_profile)

            # Process tabular features through the saved pipeline
            tab_df = pd.DataFrame([tab_profile])
            tab_processed = preprocessor.transform(tab_df)
            tab_processed = poly.transform(tab_processed)
            tab_tensor = torch.tensor(tab_processed, dtype=torch.float32).squeeze(0)

            # Extract audio embedding via HeAR
            spec_tensor = audio_segment_to_spectrogram(segment_audio, sr).to(device)
            with torch.no_grad():
                outputs = hear_model(spec_tensor)
                emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)

            all_tab_features.append(tab_tensor)
            all_aud_embeddings.append(emb.cpu().squeeze(0))
            all_labels.append(fusion_label)
            all_patient_ids.append(pid)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(wav_files)} files ({len(all_labels)} paired segments)")

    print(f"\nExtraction complete: {len(all_labels)} paired segments ({skipped} skipped)")
    print(f"  Positive (at-risk): {sum(all_labels)} | Negative (low-risk): {len(all_labels) - sum(all_labels)}")

    # --- Patient-level train/test split ---
    X_tab = torch.stack(all_tab_features)
    X_aud = torch.stack(all_aud_embeddings)
    y_all = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    pids = np.array(all_patient_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_tab, y_all, groups=pids))

    X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
    X_aud_train, X_aud_test = X_aud[train_idx], X_aud[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    train_patients = set(pids[train_idx])
    test_patients = set(pids[test_idx])
    print(f"\nPatient-level split:")
    print(f"  Train: {len(X_tab_train)} segments from {len(train_patients)} patients")
    print(f"  Test:  {len(X_tab_test)} segments from {len(test_patients)} patients")
    print(f"  Patient overlap: {len(train_patients & test_patients)} (should be 0)")
    print(f"  Train positive: {int(y_train.sum())} | Train negative: {len(y_train) - int(y_train.sum())}")
    print(f"  Test  positive: {int(y_test.sum())} | Test  negative: {len(y_test) - int(y_test.sum())}")

    train_dataset = TensorDataset(X_tab_train, X_aud_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_tab_test, X_aud_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize fusion model ---
    tab_ckpt = torch.load('best_model.pth', map_location='cpu', weights_only=False)
    tab_input_dim = tab_ckpt['input_dim']

    print(f"\nInitializing MultimodalFusionNet (tab_dim={tab_input_dim})...")
    model = MultimodalFusionNet(
        tab_input_dim=tab_input_dim,
        tab_weights_path='best_model.pth',
        aud_weights_path='best_respiratory_classifier.pth'
    ).to(device)

    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.fusion_head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Training loop ---
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_test_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    print("\nStarting Fusion Training...")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        preds, targets = [], []
        for tab_x, aud_x, y in train_loader:
            tab_x, aud_x, y = tab_x.to(device), aud_x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(tab_x, aud_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds.append((torch.sigmoid(logits) >= 0.5).cpu().int())
            targets.append(y.cpu().int())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Evaluate
        model.eval()
        total_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for tab_x, aud_x, y in test_loader:
                tab_x, aud_x, y = tab_x.to(device), aud_x.to(device), y.to(device)
                logits = model(tab_x, aud_x)
                loss = criterion(logits, y)
                total_loss += loss.item()
                preds.append((torch.sigmoid(logits) >= 0.5).cpu().int())
                targets.append(y.cpu().int())

        avg_test_loss = total_loss / len(test_loader)
        test_acc = accuracy_score(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        test_losses.append(avg_test_loss)
        test_accs.append(test_acc)

        scheduler.step(avg_test_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | LR: {lr:.1e}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'tab_input_dim': tab_input_dim},
                       'best_fusion_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}! Best: epoch {best_epoch}")
                break

    # --- Load best model ---
    actual_epochs = len(train_losses)
    print(f"\nLoading best fusion model from epoch {best_epoch}...")
    best_ckpt = torch.load('best_fusion_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # --- Final evaluation ---
    model.eval()
    all_logits, all_true = [], []
    with torch.no_grad():
        for tab_x, aud_x, y in test_loader:
            tab_x, aud_x = tab_x.to(device), aud_x.to(device)
            logits = model(tab_x, aud_x)
            all_logits.append(logits.cpu())
            all_true.append(y)

    all_logits = torch.cat(all_logits)
    all_true = torch.cat(all_true).numpy().astype(int)
    all_probs = torch.sigmoid(all_logits).numpy()

    # Threshold tuning via Youden's J statistic
    fpr, tpr, thresholds_roc = roc_curve(all_true, all_probs)
    j_scores = tpr - fpr
    best_t_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds_roc[best_t_idx])

    default_preds = (all_probs >= 0.5).astype(int)
    optimal_preds = (all_probs >= optimal_threshold).astype(int)

    print(f"\n--- Threshold Tuning ---")
    print(f"  Default (0.50):  Acc={accuracy_score(all_true, default_preds):.4f} | F1={f1_score(all_true, default_preds):.4f}")
    print(f"  Optimal ({optimal_threshold:.4f}): Acc={accuracy_score(all_true, optimal_preds):.4f} | F1={f1_score(all_true, optimal_preds):.4f}")

    all_preds = optimal_preds
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    auc = roc_auc_score(all_true, all_probs)

    print(f"\n{'='*55}")
    print(f"  FUSION MODEL — FINAL TEST RESULTS (threshold={optimal_threshold:.4f})")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"{'='*55}")
    print(f"\nClassification Report:\n")
    print(classification_report(all_true, all_preds, target_names=['Low-Risk', 'At-Risk']))

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multimodal Fusion Classifier — Training Results', fontsize=14, fontweight='bold')
    epochs_range = range(1, actual_epochs + 1)

    axes[0, 0].plot(epochs_range, train_losses, 'b-o', label='Train Loss', markersize=4)
    axes[0, 0].plot(epochs_range, test_losses, 'r-o', label='Test Loss', markersize=4)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, train_accs, 'b-o', label='Train Accuracy', markersize=4)
    axes[0, 1].plot(epochs_range, test_accs, 'r-o', label='Test Accuracy', markersize=4)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_true, all_preds)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix'); axes[1, 0].set_xlabel('Predicted'); axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Low-Risk', 'At-Risk']); axes[1, 0].set_yticklabels(['Low-Risk', 'At-Risk'])
    for ci in range(2):
        for cj in range(2):
            axes[1, 0].text(cj, ci, str(cm[ci, cj]), ha='center', va='center',
                          color='white' if cm[ci, cj] > cm.max() / 2 else 'black', fontsize=16)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    fpr_plot, tpr_plot, _ = roc_curve(all_true, all_probs)
    axes[1, 1].plot(fpr_plot, tpr_plot, 'b-', label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].plot(fpr[best_t_idx], tpr[best_t_idx], 'ro', markersize=10,
                    label=f'Optimal threshold = {optimal_threshold:.3f}')
    axes[1, 1].set_xlabel('False Positive Rate'); axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'fusion_training_results.png'
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nPlots saved to '{plot_path}'")

    # --- Save final model + config ---
    torch.save({'model_state_dict': model.state_dict(), 'tab_input_dim': tab_input_dim},
               'multimodal_fusion_final.pth')

    config = {'optimal_threshold': optimal_threshold, 'best_epoch': best_epoch, 'tab_input_dim': tab_input_dim}
    with open('fusion_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Fusion model saved to 'multimodal_fusion_final.pth'")
    print(f"Config saved to 'fusion_config.json'")

if __name__ == "__main__":
    main()