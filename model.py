import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from config import * 

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class SpatialProcessor(nn.Module):
    def __init__(self, mid_dim, num_nodes, spatial_type='mlp', adj_matrix=None):
        super().__init__()
        self.spatial_type = spatial_type
        
        if spatial_type == 'mlp':
            self.processor = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                nn.ReLU()
            )
        elif spatial_type == 'cnn':
            self.processor = nn.Sequential(
                nn.Conv2d(mid_dim, mid_dim, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(mid_dim),
                nn.ReLU()
            )
        elif spatial_type == 'gcn':
            if adj_matrix is None:
                raise ValueError("adj_matrix is required for GCN")
            adj_matrix = torch.FloatTensor(adj_matrix)
            D = torch.diag(1.0 / torch.sqrt(torch.sum(adj_matrix + torch.eye(num_nodes), dim=1)))
            self.register_buffer('adj', torch.matmul(torch.matmul(D, adj_matrix + torch.eye(num_nodes)), D))
            self.processor = GraphConvolution(mid_dim, mid_dim)
    
    def forward(self, x):
        """
        输入: [B, T, N, D] 或 [B, 1, N, D]
        输出: 保持相同shape
        """
        if self.spatial_type == 'mlp':
            return self.processor(x)
        elif self.spatial_type == 'cnn':
            B, T, N, D = x.shape
            x = x.reshape(B*T, D, N, 1)
            x = self.processor(x)
            return x.reshape(B, T, N, D)
        else:  # gcn
            B, T, N, D = x.shape
            x = x.reshape(B*T, N, D)
            x = self.processor(x, self.adj)
            return x.reshape(B, T, N, D)

class TemporalProcessor(nn.Module):
    def __init__(self, mid_dim, seq_len, temporal_type='mlp'):
        super().__init__()
        self.temporal_type = temporal_type
        
        if temporal_type == 'mlp':
            self.processor = nn.Sequential(
                nn.Linear(seq_len, 1),
                nn.LayerNorm(1),
                nn.ReLU()
            )
        
        elif temporal_type == 'gru':
            self.processor = nn.GRU(
                input_size=mid_dim,
                hidden_size=mid_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.proj = nn.Linear(mid_dim * 2, mid_dim)
                
        elif temporal_type == 'attention':
            self.processor = nn.TransformerEncoderLayer(
                d_model=mid_dim,
                nhead=4,
                dim_feedforward=mid_dim * 2,
                batch_first=True
            )
            self.proj = nn.Linear(seq_len, 1)
    
    def forward(self, x):
        """
        输入: [B, T, N, D]
        输出: [B, 1, N, D]
        """
        B, T, N, D = x.shape
        
        if self.temporal_type == 'mlp':
            x = x.permute(0, 2, 3, 1)  # [B, N, D, T]
            x = self.processor(x)       # [B, N, D, 1]
            x = x.permute(0, 3, 1, 2)  # [B, 1, N, D]
            
        elif self.temporal_type == 'gru':
            x = x.reshape(B * N, T, D)
            x, _ = self.processor(x)
            x = self.proj(x)
            x = x[:, -1:, :]           # [B*N, 1, D]
            x = x.reshape(B, 1, N, D)
            
        else:  # attention
            x = x.reshape(B * N, T, D)
            x = self.processor(x)
            x = x.reshape(B, N, T, D)
            x = x.permute(0, 1, 3, 2)  # [B, N, D, T]
            x = self.proj(x)           # [B, N, D, 1]
            x = x.permute(0, 3, 1, 2)  # [B, 1, N, D]
            
        return x

class Decoder(nn.Module):
    def __init__(self, mid_dim, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Sequential(
            nn.Linear(1, pred_len),
            nn.LayerNorm(pred_len)
        )
    
    def forward(self, x):
        """
        输入: [B, 1, N, D]
        输出: [B, P, N, D]
        """
        x = x.permute(0, 2, 3, 1)  # [B, N, D, 1]
        x = self.proj(x)           # [B, N, D, P]
        x = x.permute(0, 3, 1, 2)  # [B, P, N, D]
        return x

class ModularTrafficPredictor(nn.Module):
    def __init__(self, config: ModelConfig, spatial='mlp', temporal='mlp', spatial_first=True, adj_matrix=None):
        super().__init__()
        self.spatial_first = spatial_first
        self.mid_dim = config.hidden_dim // 2
        
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, self.mid_dim),
            nn.LayerNorm(self.mid_dim),
            nn.ReLU()
        )
        
        # Encoders
        self.spatial_encoder = SpatialProcessor(
            mid_dim=self.mid_dim,
            num_nodes=config.num_nodes,
            spatial_type=spatial,
            adj_matrix=adj_matrix if spatial == 'gcn' else None
        )
        
        self.temporal_encoder = TemporalProcessor(
            mid_dim=self.mid_dim,
            seq_len=config.seq_len,
            temporal_type=temporal
        )
        
        # Single Decoder
        self.decoder = Decoder(
            mid_dim=self.mid_dim,
            pred_len=config.pred_len
        )
        
        self.output_proj = nn.Linear(self.mid_dim, config.num_features)
    
    def forward(self, x):
        """
        输入: [B, T, N, F]
        输出: [B, P, N, F]
        """
        x = self.feature_proj(x)  # [B, T, N, D]
        
        if self.spatial_first:
            x = self.spatial_encoder(x)     # [B, T, N, D]
            x = self.temporal_encoder(x)    # [B, 1, N, D]
        else:
            x = self.temporal_encoder(x)    # [B, 1, N, D]
            x = self.spatial_encoder(x)     # [B, 1, N, D]
        
        x = self.decoder(x)                 # [B, P, N, D]
        return self.output_proj(x)

def test_model():
    config = Config("METR-LA")
    print(config)
    batch_size = config.batch_size
    
    x = torch.randn(config.batch_size, config.seq_len, config.num_nodes, config.num_features)
    adj = np.random.rand(config.num_nodes, config.num_nodes)
    adj = (adj + adj.T) / 2
    
    spatial_types = ['mlp', 'cnn', 'gcn']
    temporal_types = ['mlp', 'gru', 'attention']
    combinations = [
        (spatial, temporal) 
        for spatial in spatial_types
        for temporal in temporal_types
    ]
    
    for spatial, temporal in combinations:
        for spatial_first in [True, False]:
            print(f"\nTesting {spatial}-{temporal} combination (spatial_first={spatial_first}):")
            
            model = ModularTrafficPredictor(
                config, 
                spatial=spatial, 
                temporal=temporal,
                spatial_first=spatial_first,
                adj_matrix=adj if spatial == 'gcn' else None
            )
            
            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {num_params/1e6:.2f}M")
            
            y = model(x)
            expected_shape = (batch_size, config.pred_len, config.num_nodes, config.num_features)
            assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
            
            loss = y.mean()
            loss.backward()
            
            print("Test passed!")

if __name__ == "__main__":
    test_model()