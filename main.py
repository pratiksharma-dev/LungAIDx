import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import components from the newly separated files
from data_utils import load_and_process_multimodal_data, MultimodalCancerDataset
from model import MultimodalFusionMLP
from engine import train_one_epoch, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    print(f"Using device: {device}")
    print(f"Number of GPUs available: {gpu_count}")

    if gpu_count > 1:
        print("🚀 Dual-GPU Mode Activated! using DataParallel")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.backends.cudnn.benchmark = True

    (X_train_tab, X_train_aud, X_train_txt), (X_test_tab, X_test_aud, X_test_txt), y_train, y_test = load_and_process_multimodal_data()
    
    BATCH_SIZE = 2048 if gpu_count > 1 else 128
    
    train_loader = DataLoader(
        MultimodalCancerDataset(X_train_tab, X_train_aud, X_train_txt, y_train), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        MultimodalCancerDataset(X_test_tab, X_test_aud, X_test_txt, y_test), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = MultimodalFusionMLP(
        tabular_dim=X_train_tab.shape[1], 
        audio_dim=768, 
        text_dim=1024
    )

    if gpu_count > 1:
        model = nn.DataParallel(model) 
    
    model = model.to(device) 

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
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_rec, v_auc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        test_recalls.append(v_rec)
        test_aucs.append(v_auc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} | Test Loss: {v_loss:.4f} | Recall: {v_rec:.4f} | AUC: {v_auc:.4f}")

    print("Done!")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, epochs + 1), test_recalls, 'g-o', label='Test Recall', markersize=4)
    ax1.plot(range(1, epochs + 1), test_aucs, 'purple', marker='o', label='Test ROC AUC', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Screening Performance Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, epochs + 1), train_losses, 'b-o', label='Train Loss', markersize=4)
    ax2.plot(range(1, epochs + 1), test_losses, 'r-o', label='Test Loss', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Test Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multimodal_training_results.png', dpi=150)
    print("📊 Plot saved as 'multimodal_training_results.png'")

if __name__ == "__main__":
    main()