import torch
from sklearn.metrics import recall_score, roc_auc_score

def train_one_epoch(model, loader, criterion, optimizer, device):
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

def evaluate(model, loader, criterion, device):
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
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    recall = recall_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5 
        
    return total_loss / len(loader), correct / total, recall, auc