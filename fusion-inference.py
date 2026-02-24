import os
import re
import json
import glob
import random
import threading
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import librosa
import kagglehub
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

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

        tab_state = torch.load(tab_weights_path, map_location='cpu', weights_only=False)['model_state_dict']
        aud_state = torch.load(aud_weights_path, map_location='cpu', weights_only=True)

        tab_state = {k.replace('net.', 'features.'): v for k, v in tab_state.items() if 'net.15' not in k}
        aud_state = {k.replace('classifier.', 'features.'): v for k, v in aud_state.items() if 'classifier.12' not in k}

        self.tab_base.load_state_dict(tab_state, strict=False)
        self.aud_base.load_state_dict(aud_state, strict=False)

        for param in self.tab_base.parameters():
            param.requires_grad = False
        for param in self.aud_base.parameters():
            param.requires_grad = False

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
# 2. Audio helpers 
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
# 3. Tabular profile generator 
# ==========================================
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
    if rng is None:
        rng = np.random.default_rng()

    profile = DIAGNOSIS_PROFILES.get(diagnosis, DIAGNOSIS_PROFILES['Healthy'])
    age_lo, age_hi = profile['age']
    py_lo, py_hi = profile['pack_years']

    age = rng.integers(age_lo, age_hi + 1)
    pack_years = rng.uniform(py_lo, py_hi)
    copd = profile['copd_diagnosis']

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
    tab_risk = (tabular_profile['pack_years'] > 20 and
                tabular_profile['age'] > 45 and
                (tabular_profile['copd_diagnosis'] == 1 or
                 tabular_profile['family_history'] == 1))
    return 1 if (audio_label == 1 or tab_risk) else 0


# ==========================================
# 4. MedGemma Report Generator
# ==========================================
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"

def load_medgemma():
    """Load MedGemma-4B-IT for clinical report generation."""
    print("\nLoading MedGemma-4B-IT medical language model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MEDGEMMA_MODEL_ID, token=HF_TOKEN
    )
   
    # Note: eager attention used because Flash/MemEfficient SDPA kernels are
    # unavailable on Windows PyTorch + Gemma3 GQA (mismatched Q/KV heads).
    medgemma = AutoModelForCausalLM.from_pretrained(
        MEDGEMMA_MODEL_ID,
        token=HF_TOKEN,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    medgemma.eval()
    print("MedGemma loaded successfully.")
    return tokenizer, medgemma


def build_clinical_prompt(patient_info, tab_profile, segment_results, patient_summary):
    """Build a structured clinical prompt from all model outputs."""
    # Format tabular profile as readable text
    gender_str = "Male" if tab_profile['gender'] == 0 else "Female"
    radon_str = ["None", "Moderate", "High"][int(tab_profile['radon_exposure'])]
    alcohol_str = ["None", "Moderate", "Heavy"][int(tab_profile['alcohol_consumption'])]

    # Build segment summary
    symptomatic_segs = sum(1 for s in segment_results if s['audio_label'] == 1)
    at_risk_segs = sum(1 for s in segment_results if s['prediction'] == 1)
    total_segs = len(segment_results)

    seg_detail_lines = []
    for s in segment_results:
        audio_status = "crackles/wheezes detected" if s['audio_label'] == 1 else "normal breath sounds"
        risk_level = "AT-RISK" if s['prediction'] == 1 else "LOW-RISK"
        seg_detail_lines.append(
            f"  Cycle {s['seg_num']:2d} ({s['start']:.1f}s–{s['end']:.1f}s): "
            f"{audio_status} | fusion risk probability {s['prob']*100:.1f}% → {risk_level}"
        )
    seg_details = "\n".join(seg_detail_lines)

    prompt = f"""You are a senior pulmonologist and oncologist reviewing multimodal AI diagnostic outputs for a patient. Based on the following data from our LungAIDx diagnostic system, generate a comprehensive cancer risk assessment report.

IMPORTANT: This is for clinical decision support only. Structure your response as a formal medical report.

=== PATIENT DEMOGRAPHICS & HISTORY ===
- Patient ID: {patient_info['pid']}
- Known Respiratory Diagnosis: {patient_info['diagnosis']}
- Age: {int(tab_profile['age'])} years
- Gender: {gender_str}
- Smoking History: {tab_profile['pack_years']:.1f} pack-years
- COPD Diagnosis: {"Yes" if tab_profile['copd_diagnosis'] == 1 else "No"}
- Radon Exposure: {radon_str}
- Asbestos Exposure: {"Yes" if tab_profile['asbestos_exposure'] == 1 else "No"}
- Secondhand Smoke Exposure: {"Yes" if tab_profile['secondhand_smoke_exposure'] == 1 else "No"}
- Alcohol Consumption: {alcohol_str}
- Family History of Lung Cancer: {"Yes" if tab_profile['family_history'] == 1 else "No"}

=== AI MODEL OUTPUTS ===

1. HeAR Audio Analysis (Google Health Acoustic Representations):
   - Audio file analyzed: {patient_info['wav_name']}
   - Total respiratory cycles analyzed: {total_segs}
   - Cycles with detected crackles/wheezes: {symptomatic_segs}/{total_segs}

2. Neural Network Risk Fusion (Tabular + Audio):
   - Risk model prediction threshold: {patient_summary['threshold']:.4f}
   - Segments classified as At-Risk: {at_risk_segs}/{total_segs} ({at_risk_segs/total_segs*100:.1f}%)
   - Mean fusion risk probability: {patient_summary['avg_prob']*100:.2f}%
   - Overall AI prediction: {patient_summary['overall_pred']}

3. Per-Cycle Breakdown:
{seg_details}

=== INSTRUCTIONS ===
Based on ALL the above data, generate a structured report with these sections:
1. **Clinical Summary** — Brief overview of the patient and key findings
2. **Risk Factor Analysis** — Detailed assessment of each risk factor and their combined significance
3. **Respiratory Audio Findings** — Interpretation of the HeAR model acoustic analysis
4. **Integrated Risk Assessment** — Combined interpretation of tabular risk + audio findings
5. **Risk Stratification** — Classify overall risk as LOW / MODERATE / HIGH / VERY HIGH with justification
6. **Recommended Next Steps** — Specific clinical actions (screening, imaging, follow-up)
7. **Disclaimer** — Standard AI-assisted diagnostic disclaimer
"""
    return prompt


def generate_report(tokenizer, medgemma, prompt):
    """Generate clinical report using MedGemma."""
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    )
    inputs = {k: v.to(medgemma.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = medgemma.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )

    # Decode only the generated part (skip the input prompt tokens)
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    report = tokenizer.decode(generated, skip_special_tokens=True)
    return report


# ==========================================
# 5. Inference Pipeline
# ==========================================
def run_multimodal_inference():
    print(f"Multimodal Fusion Inference on {device}")
    print("=" * 60)

    # --- Load config with optimal threshold ---
    config_path = 'fusion_config.json'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        optimal_threshold = config.get('optimal_threshold', 0.5)
        print(f"Loaded optimal threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        print("No fusion_config.json found — using default threshold 0.50")

    # --- Load preprocessing pipeline ---
    preprocessor = joblib.load('preprocessor.pkl')
    poly = joblib.load('poly_features.pkl')

    # --- Load HeAR model ---
    print("\nLoading HeAR foundation model...")
    hear_model = AutoModel.from_pretrained(
        "google/hear-pytorch", token=HF_TOKEN, trust_remote_code=True
    ).to(device)
    hear_model.eval()
    for param in hear_model.parameters():
        param.requires_grad = False

    # --- Load fusion model ---
    print("Loading fusion model...")
    checkpoint = torch.load('multimodal_fusion_final.pth', map_location=device, weights_only=False)
    model = MultimodalFusionNet(
        tab_input_dim=checkpoint['tab_input_dim'],
        tab_weights_path='best_model.pth',
        aud_weights_path='best_respiratory_classifier.pth'
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load ICBHI dataset ---
    print("\nLoading ICBHI respiratory sound database...")
    audio_path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    audio_dir = os.path.join(audio_path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
    diag_csv = os.path.join(audio_path, "Respiratory_Sound_Database", "Respiratory_Sound_Database", "patient_diagnosis.csv")

    patient_diag = {}
    diag_df = pd.read_csv(diag_csv, header=None, names=['patient_id', 'diagnosis'])
    for _, row in diag_df.iterrows():
        patient_diag[int(row['patient_id'])] = row['diagnosis'].strip()

    # --- Pick a random audio file ---
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    random_wav = random.choice(wav_files)
    txt_path = random_wav.replace('.wav', '.txt')

    pid = get_patient_id(random_wav)
    diagnosis = patient_diag.get(pid, 'Unknown')
    wav_name = os.path.basename(random_wav)

    print(f"\n{'='*60}")
    print(f"  SELECTED PATIENT")
    print(f"{'='*60}")
    print(f"  Audio File  : {wav_name}")
    print(f"  Patient ID  : {pid}")
    print(f"  Diagnosis   : {diagnosis}")

    # --- Parse respiratory cycle segments ---
    segments = parse_segments(txt_path) if os.path.exists(txt_path) else []
    if not segments:
        print("  No annotated segments found for this file.")
        return

    print(f"  Segments    : {len(segments)} respiratory cycles")

    # --- Generate tabular profile for this patient ---
    rng = np.random.default_rng()
    tab_profile = generate_tabular_profile(diagnosis, rng)

    # Pretty-print the tabular profile
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC TABULAR PROFILE (based on {diagnosis} diagnosis)")
    print(f"{'='*60}")
    profile_labels = {
        'age': 'Age', 'gender': 'Gender', 'pack_years': 'Pack Years',
        'radon_exposure': 'Radon Exposure', 'asbestos_exposure': 'Asbestos Exposure',
        'secondhand_smoke_exposure': 'Secondhand Smoke', 'copd_diagnosis': 'COPD Diagnosis',
        'alcohol_consumption': 'Alcohol Consumption', 'family_history': 'Family History',
    }
    for key, label in profile_labels.items():
        val = tab_profile[key]
        if key == 'gender':
            print(f"  {label:25s}: {'Male' if val == 0 else 'Female'}")
        elif key in ('asbestos_exposure', 'secondhand_smoke_exposure', 'copd_diagnosis', 'family_history'):
            print(f"  {label:25s}: {'Yes' if val == 1 else 'No'}")
        elif key == 'radon_exposure':
            print(f"  {label:25s}: {['None', 'Moderate', 'High'][int(val)]}")
        elif key == 'alcohol_consumption':
            print(f"  {label:25s}: {['None', 'Moderate', 'Heavy'][int(val)]}")
        elif key == 'pack_years':
            print(f"  {label:25s}: {val:.1f}")
        else:
            print(f"  {label:25s}: {int(val)}")

    # --- Process tabular through the saved pipeline ---
    tab_df = pd.DataFrame([tab_profile])
    tab_processed = preprocessor.transform(tab_df)
    tab_processed = poly.transform(tab_processed)
    tab_tensor = torch.tensor(tab_processed, dtype=torch.float32).to(device)

    # --- Load audio ---
    try:
        full_audio, sr = librosa.load(random_wav, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  Error loading audio: {e}")
        return

    # --- Per-segment fusion inference ---
    print(f"\n{'='*60}")
    print(f"  SEGMENT-LEVEL FUSION ANALYSIS")
    print(f"{'='*60}")
    print(f"  {'Seg':>3s} | {'Time':>12s} | {'Audio':>8s} | {'Fusion Prob':>11s} | {'Prediction':>10s} | {'Ground Truth':>12s}")
    print(f"  {'---':>3s}-+-{'---':>12s}-+-{'---':>8s}-+-{'---':>11s}-+-{'---':>10s}-+-{'---':>12s}")

    seg_probs = []
    seg_preds = []
    seg_ground_truths = []

    for seg_idx, (start, end, audio_label) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = full_audio[start_sample:end_sample]

        if len(segment_audio) < sr * 0.3:
            continue

        # Get audio embedding
        spec_tensor = audio_segment_to_spectrogram(segment_audio, sr).to(device)
        with torch.no_grad():
            outputs = hear_model(spec_tensor)
            emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)

        # Fusion forward pass
        with torch.no_grad():
            logit = model(tab_tensor, emb)
            prob = torch.sigmoid(logit).item()

        prediction = 1 if prob >= optimal_threshold else 0
        ground_truth = compute_fusion_label(audio_label, tab_profile)

        seg_probs.append(prob)
        seg_preds.append(prediction)
        seg_ground_truths.append(ground_truth)

        audio_str = "Symptom" if audio_label == 1 else "Normal"
        pred_str = "At-Risk" if prediction == 1 else "Low-Risk"
        gt_str = "At-Risk" if ground_truth == 1 else "Low-Risk"

        print(f"  {seg_idx+1:3d} | {start:5.1f}s-{end:5.1f}s | {audio_str:>8s} | {prob:>10.4f}  | {pred_str:>10s} | {gt_str:>12s}")

    # --- Patient summary ---
    if not seg_probs:
        print("\n  No valid segments to analyze.")
        return

    avg_prob = np.mean(seg_probs)
    at_risk_count = sum(seg_preds)
    total_segs = len(seg_preds)
    correct = sum(p == g for p, g in zip(seg_preds, seg_ground_truths))
    seg_accuracy = correct / total_segs

    overall_pred = "At-Risk" if avg_prob >= optimal_threshold else "Low-Risk"

    print(f"\n{'='*60}")
    print(f"  PATIENT SUMMARY — {wav_name}")
    print(f"{'='*60}")
    print(f"  Patient ID              : {pid}")
    print(f"  Known Diagnosis         : {diagnosis}")
    print(f"  Total Segments Analyzed : {total_segs}")
    print(f"  At-Risk Segments        : {at_risk_count}/{total_segs} ({at_risk_count/total_segs*100:.1f}%)")
    print(f"  Segment Accuracy        : {correct}/{total_segs} ({seg_accuracy*100:.1f}%)")
    print(f"  Mean Risk Probability   : {avg_prob*100:.2f}%")
    print(f"  Optimal Threshold       : {optimal_threshold:.4f}")
    print(f"  Overall Prediction      : {overall_pred}")
    print(f"{'='*60}")

    # =============================================
    # MedGemma Clinical Report Generation
    # =============================================
    print(f"\n{'='*60}")
    print(f"  GENERATING CLINICAL REPORT VIA MedGemma")
    print(f"{'='*60}")

    # Free up GPU memory from HeAR before loading MedGemma
    del hear_model
    torch.cuda.empty_cache()

    tokenizer, medgemma = load_medgemma()

    # Collect structured segment results for the prompt
    segment_results = []
    seg_idx_offset = 0
    for seg_idx, (start, end, audio_label) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = full_audio[start_sample:end_sample]
        if len(segment_audio) < sr * 0.3:
            continue
        if seg_idx_offset < len(seg_probs):
            segment_results.append({
                'seg_num': seg_idx_offset + 1,
                'start': start,
                'end': end,
                'audio_label': audio_label,
                'prob': seg_probs[seg_idx_offset],
                'prediction': seg_preds[seg_idx_offset],
            })
            seg_idx_offset += 1

    patient_info = {
        'pid': pid,
        'diagnosis': diagnosis,
        'wav_name': wav_name,
    }
    patient_summary = {
        'avg_prob': avg_prob,
        'threshold': optimal_threshold,
        'overall_pred': overall_pred,
    }

    clinical_prompt = build_clinical_prompt(
        patient_info, tab_profile, segment_results, patient_summary
    )

    print("Generating report...\n")
    report = generate_report(tokenizer, medgemma, clinical_prompt)

    print(f"{'='*60}")
    print(f"  LungAIDx — AI-GENERATED CANCER RISK REPORT")
    print(f"{'='*60}")
    print(report)
    print(f"{'='*60}")
    print(f"  END OF REPORT")
    print(f"{'='*60}")

    # Cleanup
    del medgemma, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_multimodal_inference()