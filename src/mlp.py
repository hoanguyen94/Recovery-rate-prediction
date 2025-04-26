import torch.nn as nn
from typing import List
from torch.utils.data import Dataset
import torch

class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
        activation_name='ReLU',
        negative_slope=0.01
    ) -> None:
        super().__init__()

        self.linear_layers = nn.ModuleList([
            nn.Linear(d_layers[i - 1] if i else d_in, x)
            for i, x in enumerate(d_layers)
            ])
        
        # self.dropout_layer = nn.ModuleList([nn.Dropout(p) for p in dropout])
        self.dropout_layer = nn.Dropout(dropout)
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.activation_layer = getattr(nn, activation_name)()
        
        if self.activation_layer == nn.LeakyReLU:
            self.activation_layer = self.activation_layer(negative_slope)
     

    def forward(self, inputs):

        x = inputs
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = self.activation_layer(x)
            x = self.dropout_layer(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)