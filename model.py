import torch
import torch.nn as nn

class MultimodalFusionMLP(nn.Module):
    def __init__(self, tabular_dim, audio_dim, text_dim, hidden_dim=512):
        super(MultimodalFusionMLP, self).__init__()
        input_dim = tabular_dim + audio_dim + text_dim
        
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
        fused = torch.cat((tab, aud, txt), dim=1)
        return self.net(fused)