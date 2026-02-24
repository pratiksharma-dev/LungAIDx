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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import kagglehub 

# ==========================================
# 1. Multi-GPU Setup
# ==========================================
# This will now detect both 3090s
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {gpu_count}")

    if gpu_count > 1:
        print("🚀 Dual-GPU Mode Activated! using DataParallel")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Optimization for fixed input sizes (common in tabular data)
        torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Data Loading (Same as before)
# ==========================================
def load_and_process_data():
    print("\n--- Downloading Data via KaggleHub ---")
    path = kagglehub.dataset_download("mikeytracegod/lung-cancer-risk-dataset")
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)

    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    target_col = 'lung_cancer'
    # Convert target column to numeric (handles 'Yes'/'No', 'yes'/'no', 1/0, etc.)
    # Always convert to string first, then map to numeric
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = df[target_col].map({'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
        
    y = df[target_col].values.astype(np.float32)
    X = df.drop(columns=[target_col])

    # Ordinal encode specific categorical levels
    ordinal_map_1 = {'none': 0, 'moderate': 1, 'heavy': 2}
    ordinal_map_2 = {'low': 0, 'medium': 1, 'high': 2}
    gender_map = {'male': 0, 'female': 1}
    yesno_map = {'no': 0, 'yes': 1}
    for col in X.select_dtypes(include=['object', 'string']).columns:
        # Normalize: fill NaN/NA with 'none', then lowercase
        lower_vals = X[col].fillna('none').astype(str).str.strip().str.lower()
        # Also catch any string representations of missing values
        lower_vals = lower_vals.replace({'nan': 'none', '<na>': 'none'})
        unique_vals = set(lower_vals.unique())
        if unique_vals <= set(ordinal_map_1.keys()):
            X[col] = lower_vals.map(ordinal_map_1).astype(np.float64)
        elif unique_vals <= set(ordinal_map_2.keys()):
            X[col] = lower_vals.map(ordinal_map_2).astype(np.float64)
        elif unique_vals <= set(gender_map.keys()):
            X[col] = lower_vals.map(gender_map).astype(np.float64)
        elif unique_vals <= set(yesno_map.keys()):
            X[col] = lower_vals.map(yesno_map).astype(np.float64)

    print(X)

    # Build preprocessor — only include 'cat' if object columns remain
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'string']).columns
    transformers = [('num', StandardScaler(), num_cols)]
    if len(cat_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    preprocessor = ColumnTransformer(transformers)

    X_processed = preprocessor.fit_transform(X)

    # Feature engineering: add pairwise interaction features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_processed = poly.fit_transform(X_processed)
    print(f"Features after interactions: {X_processed.shape[1]}")

    # Save preprocessing pipeline for inference
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(poly, 'poly_features.pkl')
    print("Saved preprocessor.pkl and poly_features.pkl")

   # 40k / 10k split
    indices = np.random.permutation(len(X_processed))
    # Ensure we don't go out of bounds if dataset size changes
    limit = min(len(indices), 50000) 
    split_point = int(limit * 0.8) # 80% train

    X_train, y_train = X_processed[indices[:split_point]], y[indices[:split_point]]
    X_test, y_test = X_processed[indices[split_point:limit]], y[indices[split_point:limit]]

    return X_train, X_test, y_train, y_test

# ==========================================
# 3. Dataset & Model
# ==========================================
class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))

class LungRiskNet(nn.Module):
    def __init__(self, input_dim):
        super(LungRiskNet, self).__init__()
        self.net = nn.Sequential(
            # Wider entry layer to handle interaction features
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            # Residual blocks for deeper learning
            ResidualBlock(512, dropout=0.3),
            ResidualBlock(512, dropout=0.3),

            # Narrowing layers
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            ResidualBlock(256, dropout=0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Training Loop
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Sigmoid to get probability
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    return total_loss / len(loader), correct / total

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data()
    
    # --- BATCH SIZE: Smaller batches generalize better on tabular data ---
    BATCH_SIZE = 512 if gpu_count > 1 else 128
    
    train_loader = DataLoader(
        CancerDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        CancerDataset(X_test, y_test), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Initialize Model
    model = LungRiskNet(input_dim=X_train.shape[1])

    # --- KEY CHANGE: DATA PARALLEL ---
    if gpu_count > 1:
        model = nn.DataParallel(model) # Wraps the model to use multiple GPUs
    
    model = model.to(device) # Send wrapped model to main device (cuda:0)

    # Compute class weight to handle imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    print(f"Class balance — Positive: {pos_count:.0f}, Negative: {neg_count:.0f}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()  # No pos_weight — classes are near-balanced (55/45)

    epochs = 80
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=epochs
    )
    print(f"\n--- Starting Training on {gpu_count} GPUs with Batch Size {BATCH_SIZE} ---")
    
    # Track metrics for plotting
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        
        # Store metrics
        train_accuracies.append(t_acc * 100)
        test_accuracies.append(v_acc * 100)
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        
        # Save best model
        if v_acc > best_test_acc:
            best_test_acc = v_acc
            best_epoch = epoch + 1
            # Unwrap DataParallel if needed
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'input_dim': X_train.shape[1],
                'epoch': best_epoch,
                'test_acc': best_test_acc,
                'test_loss': v_loss,
            }, 'best_model.pth')
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2%} | Test Loss: {v_loss:.4f} Acc: {v_acc:.2%} | LR: {current_lr:.6f}")

    print(f"Done! Best Test Acc: {best_test_acc:.2%} at Epoch {best_epoch}")
    print(f"Model saved to best_model.pth")
    
    # ==========================================
    # 6. Gradient Boosting Comparison
    # ==========================================
    print("\n--- Gradient Boosting Baseline (for comparison) ---")
    gb = GradientBoostingClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_test_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"GradientBoosting | Train Acc: {gb_train_acc:.2%} | Test Acc: {gb_test_acc:.2%}")
    print(f"Neural Network   | Best Test Acc: {max(test_accuracies):.2f}%")
    
    
    # ==========================================
    # 6. Plot Training Results
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    ax1.plot(range(1, epochs + 1), train_accuracies, 'b-o', label='Train Accuracy', markersize=4)
    ax1.plot(range(1, epochs + 1), test_accuracies, 'r-o', label='Test Accuracy', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training vs Test Accuracy')
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
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    print("📊 Plot saved as 'training_results.png'")

    # ==========================================
    # 8. Save Final Checkpoint (everything needed for inference)
    # ==========================================
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'input_dim': X_train.shape[1],
        'epochs_trained': epochs,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
    }, 'final_model.pth')
    print("Saved final_model.pth")
    print("\nFiles saved for deployment:")
    print("  - best_model.pth       (best checkpoint weights)")
    print("  - final_model.pth      (final epoch weights)")
    print("  - preprocessor.pkl     (StandardScaler pipeline)")
    print("  - poly_features.pkl    (polynomial feature transformer)") 