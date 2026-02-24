"""Quick script to inspect the transformed data and diagnose issues."""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import kagglehub

print("--- Downloading Data via KaggleHub ---")
path = kagglehub.dataset_download("mikeytracegod/lung-cancer-risk-dataset")

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, csv_files[0])
df = pd.read_csv(csv_path)

if 'patient_id' in df.columns:
    df = df.drop(columns=['patient_id'])

target_col = 'lung_cancer'
df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[target_col] = df[target_col].map({'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})

y = df[target_col].values.astype(np.float32)
X = df.drop(columns=[target_col])

print("\n=== BEFORE encoding ===")
print(f"Shape: {X.shape}")
print(f"\nColumn dtypes:\n{X.dtypes}")
print(f"\nObject/string columns: {list(X.select_dtypes(include=['object', 'string']).columns)}")
print(f"Numeric columns: {list(X.select_dtypes(include=['int64', 'float64']).columns)}")

# Show unique values for each object column
for col in X.select_dtypes(include=['object', 'string']).columns:
    print(f"\n  '{col}' unique values: {X[col].str.strip().str.lower().unique()[:20]}")

# Apply ordinal encoding
ordinal_map_1 = {'none': 0, 'moderate': 1, 'heavy': 2}
ordinal_map_2 = {'low': 0, 'medium': 1, 'high': 2}
gender_map = {'male': 0, 'female': 1}
yesno_map = {'no': 0, 'yes': 1}
for col in X.select_dtypes(include=['object', 'string']).columns:
    lower_vals = X[col].fillna('none').astype(str).str.strip().str.lower()
    lower_vals = lower_vals.replace({'nan': 'none', '<na>': 'none'})
    unique_vals = set(lower_vals.unique())
    if unique_vals <= set(ordinal_map_1.keys()):
        X[col] = lower_vals.map(ordinal_map_1).astype(np.float64)
        print(f"  Mapped '{col}' with ordinal_map_1 (none/moderate/heavy)")
    elif unique_vals <= set(ordinal_map_2.keys()):
        X[col] = lower_vals.map(ordinal_map_2).astype(np.float64)
        print(f"  Mapped '{col}' with ordinal_map_2 (low/medium/high)")
    elif unique_vals <= set(gender_map.keys()):
        X[col] = lower_vals.map(gender_map).astype(np.float64)
        print(f"  Mapped '{col}' with gender_map (male/female)")
    elif unique_vals <= set(yesno_map.keys()):
        X[col] = lower_vals.map(yesno_map).astype(np.float64)
        print(f"  Mapped '{col}' with yesno_map (no/yes)")
    else:
        print(f"  WARNING: '{col}' NOT mapped. Unique vals: {unique_vals}")

print("\n=== AFTER encoding ===")
print(f"\nColumn dtypes:\n{X.dtypes}")
print(f"\nRemaining object/string columns: {list(X.select_dtypes(include=['object', 'string']).columns)}")
print(f"Numeric columns: {list(X.select_dtypes(include=['int64', 'float64']).columns)}")

# Check for NaN values
nan_counts = X.isnull().sum()
if nan_counts.any():
    print(f"\n*** WARNING: NaN values found! ***")
    print(nan_counts[nan_counts > 0])
else:
    print("\nNo NaN values found.")

# Check target
print(f"\n=== TARGET ===")
print(f"y shape: {y.shape}, dtype: {y.dtype}")
print(f"y unique values: {np.unique(y)}")
nan_in_y = np.isnan(y).sum()
print(f"NaN in y: {nan_in_y}")

# Now test the preprocessor
remaining_obj_cols = X.select_dtypes(include=['object', 'string']).columns
remaining_num_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\n=== PREPROCESSOR INPUT ===")
print(f"Numeric cols for StandardScaler ({len(remaining_num_cols)}): {list(remaining_num_cols)}")
print(f"Object cols for OneHotEncoder ({len(remaining_obj_cols)}): {list(remaining_obj_cols)}")

if len(remaining_obj_cols) == 0:
    print("\n*** NOTE: No object columns remain for OneHotEncoder. ***")
    print("*** The 'cat' transformer in ColumnTransformer will receive empty input. ***")
    print("*** This may cause issues or warnings. Consider removing the 'cat' transformer. ***")

# Build and apply preprocessor
transformers = [('num', StandardScaler(), remaining_num_cols)]
if len(remaining_obj_cols) > 0:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), remaining_obj_cols))

preprocessor = ColumnTransformer(transformers)
X_processed = preprocessor.fit_transform(X)

print(f"\n=== PROCESSED DATA ===")
print(f"X_processed shape: {X_processed.shape}")
print(f"X_processed dtype: {X_processed.dtype}")
print(f"NaN in X_processed: {np.isnan(X_processed).sum()}")
print(f"Inf in X_processed: {np.isinf(X_processed).sum()}")
print(f"\nFirst 3 rows:\n{X_processed[:3]}")
print(f"\nStats per feature (min/max/mean):")
for i in range(X_processed.shape[1]):
    col = X_processed[:, i]
    print(f"  Feature {i}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

print(f"\n=== SUMMARY ===")
print(f"Input features: {X_processed.shape[1]}")
print(f"Samples: {X_processed.shape[0]}")
print(f"Ready for neural network: {'YES' if not np.isnan(X_processed).any() and not np.isinf(X_processed).any() else 'NO - contains NaN/Inf!'}")
