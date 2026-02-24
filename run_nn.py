import os
import torch
import joblib
import numpy as np
import pandas as pd
import kagglehub
from train import LungRiskNet

# ==========================================
# 1. Load the dataset (same source as training)
# ==========================================
print("--- Loading dataset ---")
path = kagglehub.dataset_download("mikeytracegod/lung-cancer-risk-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, csv_files[0])
df = pd.read_csv(csv_path)

if 'patient_id' in df.columns:
    df = df.drop(columns=['patient_id'])

# Pick a random row
idx = np.random.randint(0, len(df))
sample_row = df.iloc[[idx]].copy()
print(f"\n--- Random Sample (Row {idx}) ---")
print(sample_row.to_string(index=False))

# ==========================================
# 2. Get actual label
# ==========================================
target_col = 'lung_cancer'
actual_raw = sample_row[target_col].values[0]
actual_label = str(actual_raw).strip().lower()
actual_numeric = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}.get(actual_label, actual_label)
print(f"\nActual label: {actual_raw} -> {actual_numeric} ({'Cancer' if actual_numeric == 1 else 'No Cancer'})")

# ==========================================
# 3. Preprocess the sample (same as train.py)
# ==========================================
X_sample = sample_row.drop(columns=[target_col])

# Apply ordinal encoding (must match train.py exactly)
ordinal_map_1 = {'none': 0, 'moderate': 1, 'heavy': 2}
ordinal_map_2 = {'low': 0, 'medium': 1, 'high': 2}
gender_map = {'male': 0, 'female': 1}
yesno_map = {'no': 0, 'yes': 1}
for col in X_sample.select_dtypes(include=['object', 'string']).columns:
    lower_vals = X_sample[col].fillna('none').astype(str).str.strip().str.lower()
    lower_vals = lower_vals.replace({'nan': 'none', '<na>': 'none'})
    unique_vals = set(lower_vals.unique())
    if unique_vals <= set(ordinal_map_1.keys()):
        X_sample[col] = lower_vals.map(ordinal_map_1).astype(np.float64)
    elif unique_vals <= set(ordinal_map_2.keys()):
        X_sample[col] = lower_vals.map(ordinal_map_2).astype(np.float64)
    elif unique_vals <= set(gender_map.keys()):
        X_sample[col] = lower_vals.map(gender_map).astype(np.float64)
    elif unique_vals <= set(yesno_map.keys()):
        X_sample[col] = lower_vals.map(yesno_map).astype(np.float64)

# ==========================================
# 4. Apply saved preprocessing pipeline
# ==========================================
preprocessor = joblib.load('preprocessor.pkl')
poly = joblib.load('poly_features.pkl')

X_processed = preprocessor.transform(X_sample)
X_processed = poly.transform(X_processed)

# ==========================================
# 5. Load model and predict
# ==========================================
checkpoint = torch.load('best_model.pth', map_location='cpu')
model = LungRiskNet(input_dim=checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

X_tensor = torch.tensor(X_processed, dtype=torch.float32)
with torch.no_grad():
    logit = model(X_tensor)
    prob = torch.sigmoid(logit).item()

predicted = 1 if prob > 0.5 else 0

# ==========================================
# 6. Compare
# ==========================================
print(f"\n{'='*40}")
print(f"  Model Probability : {prob:.4f}")
print(f"  Predicted         : {predicted} ({'Cancer' if predicted == 1 else 'No Cancer'})")
print(f"  Actual            : {actual_numeric} ({'Cancer' if actual_numeric == 1 else 'No Cancer'})")
print(f"  Match             : {'CORRECT' if predicted == actual_numeric else 'INCORRECT'}")
print(f"{'='*40}")
print(f"\n(Model from epoch {checkpoint['epoch']} with test acc: {checkpoint['test_acc']:.2%})")