import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import kagglehub 

def get_hear_embeddings(num_samples):
    return np.random.randn(num_samples, 768).astype(np.float32)

def get_medgemma_embeddings(num_samples):
    return np.random.randn(num_samples, 1024).astype(np.float32)

def load_and_process_multimodal_data():
    print("\n--- Downloading Data via KaggleHub ---")
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
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.select_dtypes(include=['object', 'string']).columns)
    ])

    X_processed_tab = preprocessor.fit_transform(X).astype(np.float32)
    num_samples = len(X_processed_tab)
    X_processed_aud = get_hear_embeddings(num_samples)
    X_processed_txt = get_medgemma_embeddings(num_samples)

    indices = np.random.permutation(num_samples)
    limit = min(len(indices), 50000) 
    split_point = int(limit * 0.8)

    train_idx, test_idx = indices[:split_point], indices[split_point:limit]

    return (X_processed_tab[train_idx], X_processed_aud[train_idx], X_processed_txt[train_idx]), \
           (X_processed_tab[test_idx], X_processed_aud[test_idx], X_processed_txt[test_idx]), \
           y[train_idx], y[test_idx]

class MultimodalCancerDataset(Dataset):
    def __init__(self, X_tab, X_aud, X_txt, y):
        self.X_tab = torch.tensor(X_tab)
        self.X_aud = torch.tensor(X_aud)
        self.X_txt = torch.tensor(X_txt)
        self.y = torch.tensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_tab[idx], self.X_aud[idx], self.X_txt[idx], self.y[idx]