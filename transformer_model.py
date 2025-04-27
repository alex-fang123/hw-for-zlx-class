import torch
import torch.nn as nn
import math

# Note: This model takes the same flattened input as the DNN.
# It treats the flattened feature vector as a single-element sequence after projection.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerForTimeSeries(nn.Module):
    # Expect flattened input_dim
    def __init__(self, input_dim, d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 特征处理层
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, input_dim] - Flattened features.
            
        Returns:
            output: Tensor, shape [batch_size, 1] - Predicted return.
        """
        # 特征投影
        src = self.feature_projection(src)  # [batch_size, d_model]
        
        # 添加序列维度
        src = src.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        memory = self.transformer_encoder(src)  # [batch_size, 1, d_model]
        
        # 移除序列维度
        memory = memory.squeeze(1)  # [batch_size, d_model]
        
        # 输出层
        output = self.output_layer(memory)  # [batch_size, 1]
        
        return output 