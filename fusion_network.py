import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Smaller branch architectures
        self.numeric_branch = nn.Sequential(
            nn.Linear(2, 16),             # Reduced from 16
            nn.ReLU(),
            nn.Dropout(0.2)             # Reduced dropout
        )
        
        self.type_branch = nn.Sequential(
            nn.Linear(64, 32),          # Reduced from 32
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.content_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.keyword_branch = nn.Sequential(
            nn.Linear(128, 64),         # Reduced from 64
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Simplified combined layers
        self.combined = nn.Sequential(
            nn.Linear(16+32+32+64, 128),  # Reduced capacity
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, 2)  # Final classification layer
    
    def forward(self, x):
        # Split input tensor into feature components
        numeric = x[:, :2]
        type_feat = x[:, 2:66]
        content_feat = x[:, 66:130]
        keyword_feat = x[:, 130:]
        
        # Process each feature branch
        numeric_out = self.numeric_branch(numeric)
        type_out = self.type_branch(type_feat)
        content_out = self.content_branch(content_feat)
        keyword_out = self.keyword_branch(keyword_feat)
        
        # Concatenate branch outputs
        combined = self.combined(torch.cat([numeric_out, type_out, content_out, keyword_out], dim=1))
        
        # Final classification
        return self.classifier(combined)

# Training setup would be similar to train.py but use this model
def get_model(input_dim):
    return FeatureFusionNetwork()