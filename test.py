import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, roc_auc_score
import kagglehub 

# ==========================================
# Multi-GPU Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {gpu_count}")

    if gpu_count > 1:
        print("🚀 Dual-GPU Mode Activated! using DataParallel")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Optimization for fixed input sizes to accelerate hardware execution
        torch.backends.cudnn.benchmark = True

# ==========================================
# Mock Foundation Model APIs
# ==========================================
def get_hear_embeddings(num_samples):
    # Simulates generating 768-dimensional audio embeddings for the dataset
    return np.random.randn(num_samples, 768).astype(np.float32)

def get_medgemma_embeddings(num_samples):
    # Simulates generating 1024-dimensional text embeddings for the dataset
    return np.random.randn(num_samples, 1024).astype(np.float32)

# ==========================================
# Data Loading & Multimodal Preprocessing
# ==========================================
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
    
    # Process tabular data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.select_dtypes(include=['object', 'string']).columns)
    ])

    X_processed_tab = preprocessor.fit_transform(X).astype(np.float32)
    
    # Generate synchronized mock audio and text embeddings for the dataset
    num_samples = len(X_processed_tab)
    X_processed_aud = get_hear_embeddings(num_samples)
    X_processed_txt = get_medgemma_embeddings(num_samples)

    # Shuffle and split all modalities consistently
    indices = np.random.permutation(num_samples)
    limit = min(len(indices), 50000) 
    split_point = int(limit * 0.8)

    train_idx, test_idx = indices[:split_point], indices[split_point:limit]

    X_train_tab, X_test_tab = X_processed_tab[train_idx], X_processed_tab[test_idx]
    X_train_aud, X_test_aud = X_processed_aud[train_idx], X_processed_aud[test_idx]
    X_train_txt, X_test_txt = X_processed_txt[train_idx], X_processed_txt[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return (X_train_tab, X_train_aud, X_train_txt), (X_test_tab, X_test_aud, X_test_txt), y_train, y_test

# ==========================================
# Dataset & Multimodal Network
# ==========================================
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

class MultimodalFusionMLP(nn.Module):
    def __init__(self, tabular_dim, audio_dim, text_dim, hidden_dim=512):
        super(MultimodalFusionMLP, self).__init__()
        
        input_dim = tabular_dim + audio_dim + text_dim
        
        # Wider network architecture to leverage high-VRAM dual GPUs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, tab, aud, txt):
        # Concatenate modalities along the feature dimension
        fused = torch.cat((tab, aud, txt), dim=1)
        return self.net(fused)

# ==========================================
# Training and Evaluation Loops
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for tab_batch, aud_batch, txt_batch, y_batch in loader:
        tab_batch, aud_batch, txt_batch, y_batch = (
            tab_batch.to(device), aud_batch.to(device), txt_batch.to(device), y_batch.to(device)
        )
        
        optimizer.zero_grad()
        outputs = model(tab_batch, aud_batch, txt_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for tab_batch, aud_batch, txt_batch, y_batch in loader:
            tab_batch, aud_batch, txt_batch, y_batch = (
                tab_batch.to(device), aud_batch.to(device), txt_batch.to(device), y_batch.to(device)
            )
            
            outputs = model(tab_batch, aud_batch, txt_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            # Store for Scikit-Learn metrics
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    # Calculate medical screening metrics
    recall = recall_score(all_targets, all_preds, zero_division=0)
    
    # Handle edge case where a batch might only have one class during testing
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5 
        
    return total_loss / len(loader), correct / total, recall, auc

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    (X_train_tab, X_train_aud, X_train_txt), (X_test_tab, X_test_aud, X_test_txt), y_train, y_test = load_and_process_multimodal_data()
    
    # Scale batch size for parallel processing
    BATCH_SIZE = 2048 if gpu_count > 1 else 128
    
    train_loader = DataLoader(
        MultimodalCancerDataset(X_train_tab, X_train_aud, X_train_txt, y_train), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        MultimodalCancerDataset(X_test_tab, X_test_aud, X_test_txt, y_test), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize Model dynamically tracking the processed tabular dimension
    model = MultimodalFusionMLP(
        tabular_dim=X_train_tab.shape[1], 
        audio_dim=768, 
        text_dim=1024
    )

    # Wrap the model to utilize multiple GPUs
    if gpu_count > 1:
        model = nn.DataParallel(model) 
    
    model = model.to(device) 

    # Calculate class weights for imbalanced cancer dataset
    num_positives = np.sum(y_train)
    num_negatives = len(y_train) - num_positives
    pos_weight = torch.tensor([num_negatives / (num_positives + 1e-5)]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 20
    print(f"\n--- Starting Multimodal Training on {gpu_count} GPUs with Batch Size {BATCH_SIZE} ---")
    
    train_losses, test_losses = [], []
    test_recalls, test_aucs = [], []
    
    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_rec, v_auc = evaluate(model, test_loader, criterion)
        
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        test_recalls.append(v_rec)
        test_aucs.append(v_auc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} | Test Loss: {v_loss:.4f} | Recall: {v_rec:.4f} | AUC: {v_auc:.4f}")

    print("Done!")
    
    # ==========================================
    # Plot Training Results
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Evaluation Metrics
    ax1.plot(range(1, epochs + 1), test_recalls, 'g-o', label='Test Recall (Sensitivity)', markersize=4)
    ax1.plot(range(1, epochs + 1), test_aucs, 'purple', marker='o', label='Test ROC AUC', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Screening Performance Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Loss
    ax2.plot(range(1, epochs + 1), train_losses, 'b-o', label='Train Loss', markersize=4)
    ax2.plot(range(1, epochs + 1), test_losses, 'r-o', label='Test Loss', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Test Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multimodal_training_results.png', dpi=150)
    plt.show()
    
    print("📊 Plot saved as 'multimodal_training_results.png'")