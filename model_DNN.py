import torch
import torch.nn as nn

class IndexEnhancementModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        # input_dim is the flattened dimension
        
        # 移除 Flatten 层
        # self.flatten = nn.Flatten() 
        self.layers = nn.ModuleList()
        
        # Input layer operates on flattened features
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.Dropout(0.3)) 
        
        # 隐藏层 
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(0.3))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # 期望 x 已经是展平的 [batch_size, flattened_features]
        if x.ndim > 2:
            raise ValueError(f"DNN expects flattened input (2D), got {x.ndim}D shape: {x.shape}")
            
        for layer in self.layers:
            x = layer(x)
        return x 