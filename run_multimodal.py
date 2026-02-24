import os
import json
import random
import threading
import torch
import torch.nn as nn
import librosa
import numpy as np
import kagglehub
from transformers import AutoModel

# Suppress harmless 403 from auto_conversion thread (discussions disabled on this repo)
_orig_excepthook = threading.excepthook
def _suppress_auto_conversion(args):
    if args.thread and "auto_conversion" in args.thread.name:
        return
    _orig_excepthook(args)
threading.excepthook = _suppress_auto_conversion

# Configuration parameters
HF_TOKEN = "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"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = "respiratory_classifier.pth"
CONFIG_PATH = "classifier_config.json"

CLIP_DURATION = 2.0
SAMPLE_RATE = 16000

# Architecture must match the trained model exactly
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
    """Parse ICBHI annotation file into (start, end, label) tuples."""
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
    """Convert audio clip to normalized log-mel spectrogram for HeAR ViT."""
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
    
    log_mel_tensor = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    log_mel_resized = torch.nn.functional.interpolate(
        log_mel_tensor, size=(192, 128), mode='bilinear', align_corners=False
    )
    return log_mel_resized

def main():
    print("Loading Foundation Model (HeAR)...")
    hear_model = AutoModel.from_pretrained(
        "google/hear-pytorch", 
        token=HF_TOKEN,
        trust_remote_code=True 
    ).to(device)
    hear_model.eval()

    print(f"Loading Custom Classifier from {MODEL_WEIGHTS}...")
    classifier = RespiratorySymptomClassifier().to(device)
    
    # Load weights safely into the architecture
    try:
        classifier.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: {MODEL_WEIGHTS} not found. Please run the training script first.")
        return
        
    classifier.eval()

    # Load optimal threshold from training
    threshold = 0.5  # default fallback
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            threshold = config.get("optimal_threshold", 0.5)
        print(f"Loaded optimal threshold: {threshold:.4f}")
    else:
        print(f"Warning: {CONFIG_PATH} not found, using default threshold 0.50")

    print("Locating ICBHI dataset...")
    path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    audio_dir = os.path.join(path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
    
    wav_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Select a random audio file and its matching annotation file
    random_wav = random.choice(wav_files)
    matching_txt = random_wav.replace('.wav', '.txt')
    
    print(f"\nAnalyzing File: {os.path.basename(random_wav)}")
    
    # Parse all annotated respiratory cycle segments
    segments = parse_segments(matching_txt)
    if not segments:
        print("No annotated segments found for this file.")
        return
    
    # Load full audio once
    full_audio, sr = librosa.load(random_wav, sr=SAMPLE_RATE, mono=True)
    
    print(f"Found {len(segments)} respiratory cycle segments\n")
    
    correct = 0
    total = 0
    
    try:
        for idx, (start, end, ground_truth) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = full_audio[start_sample:end_sample]
            
            if len(segment_audio) < sr * 0.3:
                continue
            
            actual_status = "Symptomatic" if ground_truth == 1 else "Normal"
            
            spec_tensor = audio_segment_to_spectrogram(segment_audio, sr).to(device)
            
            with torch.no_grad():
                outputs = hear_model(spec_tensor)
                emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
                
                logits = classifier(emb)
                probability = torch.sigmoid(logits).item() * 100
                prediction = 1 if probability > (threshold * 100) else 0
                predicted_status = "Symptomatic" if prediction == 1 else "Normal"
            
            match = "OK" if prediction == ground_truth else "MISS"
            if prediction == ground_truth:
                correct += 1
            total += 1
            
            print(f"  Segment {idx+1:2d} [{start:.2f}s - {end:.2f}s] | "
                  f"Truth: {actual_status:12s} | Pred: {predicted_status:12s} ({probability:5.1f}%) | {match}")
        
        print(f"\n--- SUMMARY ---")
        print(f"Segments analyzed: {total}")
        print(f"Correct predictions: {correct}/{total} ({correct/total*100:.1f}% accuracy)")
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")

if __name__ == "__main__":
    main()