import threading
import torch
import librosa
import numpy as np
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
hear_model = AutoModel.from_pretrained(
    "google/hear-pytorch", 
    token=HF_TOKEN,
    trust_remote_code=True 
).to(device)

   