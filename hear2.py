import os
import random
import threading
import torch
import librosa
import numpy as np
import kagglehub
from transformers import AutoModel

# Suppress harmless 403 error from auto_conversion thread
# (Google disables discussions on this repo, causing a benign failure)
_orig_excepthook = threading.excepthook
def _suppress_auto_conversion(args):
    if args.thread and "auto_conversion" in args.thread.name:
        return
    _orig_excepthook(args)
threading.excepthook = _suppress_auto_conversion

# Hugging Face Access Token (Required for gated models)
HF_TOKEN = "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"

# Configure hardware acceleration for the RTX 3090s
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Configuring execution on: {device}")

# Download and initialize the HeAR model
print("Loading HeAR model...")
hear_model = AutoModel.from_pretrained(
    "google/hear-pytorch", 
    token=HF_TOKEN,
    trust_remote_code=True 
).to(device)

# Lock the weights to use the model strictly as a feature extractor
hear_model.eval()
for param in hear_model.parameters():
    param.requires_grad = False

def preprocess_for_hear(file_path):
    # Load and resample audio to 16kHz mono
    audio_array, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Enforce exactly 2 seconds (32,000 samples)
    clip_length = sr * 2 
    if len(audio_array) > clip_length:
        audio_array = audio_array[:clip_length]
    else:
        audio_array = np.pad(audio_array, (0, clip_length - len(audio_array)))
        
    # Convert to 128-bin log-mel spectrogram (model expects 192 x 128 image)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, sr=sr, n_fft=1024, hop_length=160,
        n_mels=128, fmin=60, fmax=7800
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    
    # Resize to exact model input: 192 time frames x 128 mel bins
    # log_mel shape is (128, time_frames) -> transpose to (time, mel) then resize
    log_mel_tensor = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T,128)
    log_mel_resized = torch.nn.functional.interpolate(
        log_mel_tensor, size=(192, 128), mode='bilinear', align_corners=False
    )
    
    # Shape: (batch=1, channels=1, height=192, width=128)
    return log_mel_resized.to(device)

def main():
    # Download the ICBHI Respiratory Sound Database
    print("\nDownloading dataset...")
    path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")

    # Locate the audio files directory
    audio_dir = os.path.join(path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
    if not os.path.exists(audio_dir):
        audio_dir = path

    # Select a random .wav file
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    random_audio_file = os.path.join(audio_dir, random.choice(audio_files))
    print(f"\nSelected Audio File: {os.path.basename(random_audio_file)}")

    # Preprocess and Extract
    print("Preprocessing and passing audio to HeAR model...")
    audio_tensor = preprocess_for_hear(random_audio_file)
    
    with torch.no_grad():
        outputs = hear_model(audio_tensor)
        
        # Extract the final embedding vector from the model's hidden states
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1) 
            
    print(f"\nSuccessfully generated embeddings!")
    print(f"Tensor Shape: {embeddings.shape}")
    print(f"Device: {embeddings.device}")
    print(f"Embedding: {embeddings[0]}")

if __name__ == "__main__":
    main()