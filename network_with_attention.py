import torch
import torch.nn as nn
from config import *

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)  # Query
        K = self.key(x)    # Key
        V = self.value(x)  # Value
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        out = torch.matmul(attention_weights, V)
       
        print(f"Attention Active: Input = {x.shape}, Output = {out.shape}")
       
        return out

class PID_CNN1D(nn.Module):
    def __init__(self, 
                 n_classes: int = 6, 
                 num_of_feature: int = len(FEATURES_LIST), 
                 max_len: int = 1000,
                 kernal_size: list = [5, 3, 3, 3],
                 dropout: float = 0.5,
                 dense_size: int = 512):
        super(PID_CNN1D, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(num_of_feature, 64, kernal_size[0], padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(64), 
            nn.MaxPool1d(kernal_size[1], stride=kernal_size[1]), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
            nn.Conv1d(128, 128, kernal_size[2], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
            nn.MaxPool1d(kernal_size[2], stride=kernal_size[2]), nn.Dropout(dropout), 
            nn.Conv1d(128, 128, kernal_size[3], padding=1), nn.ReLU(),
            nn.BatchNorm1d(128), nn.AvgPool1d(kernal_size[3], stride=kernal_size[3])
        )

        # Attention Mechanism (Added)
        self.attention = SelfAttention(embed_size=128)

        # Infer output shape dynamically
        with torch.no_grad():
            dummy_data = torch.ones(1, num_of_feature, max_len)
            out_shape = self.convnet(dummy_data).view(1, -1).shape[-1]
            del dummy_data

        self.fc = nn.Sequential(
            nn.Linear(out_shape, dense_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dense_size, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input_layer):
        output = self.convnet(input_layer)  # CNN Feature Extraction
        output = output.permute(0, 2, 1)  # Convert (batch, channels, time) → (batch, time, channels)
        output = self.attention(output)  # Apply Attention
        output = output.permute(0, 2, 1)  # Convert back to (batch, channels, time)
        output = output.view(output.size()[0], -1)  # Flatten
        output = self.fc(output)  # Fully Connected Layer
        return output
    
    def _class_name(self):
        return "PIDCNN1D"