import torch
import torch.nn as nn

import torch
import torch.nn as nn
from dataclasses import dataclass
import itertools

@dataclass
class ModelConfig:
    num_nodes: int = 10
    seq_len: int = 12
    pred_len: int = 3
    num_features: int = 64
    hidden_dim: int = 128


class SpatialProcessor(nn.Module):
    def __init__(self, config: ModelConfig, processor_type: str, adj_matrix: torch.Tensor = None):
        super().__init__()
        self.type = processor_type
        self.num_nodes = config.num_nodes
        hidden_dim = config.hidden_dim
        
        if processor_type == 'mlp':
            self.processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        elif processor_type == 'cnn':
            self.processor = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
                nn.LayerNorm([hidden_dim, self.num_nodes, config.seq_len]),
                nn.ReLU()
            )
        elif processor_type == 'gcn':
            assert adj_matrix is not None, "GCN requires adjacency matrix"
            self.processor = GraphConvolution(hidden_dim, hidden_dim)
            # Process adjacency matrix
            adj_matrix = torch.FloatTensor(adj_matrix)
            adj_matrix = adj_matrix + torch.eye(self.num_nodes)
            D = torch.sum(adj_matrix, dim=1)
            D = torch.diag(1.0 / torch.sqrt(D))
            self.register_buffer('adj', torch.matmul(torch.matmul(D, adj_matrix), D))
            
    def forward(self, x):
        if self.type == 'mlp':
            return self.processor(x) + x  # Residual connection
        elif self.type == 'cnn':
            return self.processor(x) + x
        elif self.type == 'gcn':
            processed = self.processor(x, self.adj)
            return processed + x

class TemporalProcessor(nn.Module):
    def __init__(self, config: ModelConfig, processor_type: str):
        super().__init__()
        self.type = processor_type
        self.seq_len = config.seq_len
        self.hidden_dim = config.hidden_dim
        
        if processor_type == 'mlp':
            self.processor = nn.Sequential(
                nn.Linear(config.seq_len, config.seq_len // 2),
                nn.LayerNorm(config.seq_len // 2),
                nn.ReLU(),
                nn.Linear(config.seq_len // 2, config.pred_len)
            )
        elif processor_type == 'gru':
            self.processor = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        elif processor_type == 'attention':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.processor = nn.TransformerEncoder(encoder_layer, num_layers=3)
            
    def forward(self, x):
        if self.type == 'mlp':
            return self.processor(x)
        elif self.type == 'gru':
            output, _ = self.processor(x)
            return self.proj(output[:, -1:])
        elif self.type == 'attention':
            return self.processor(x)[:, -1:]

class ModularTrafficPredictor(nn.Module):
    def __init__(self, config: ModelConfig, spatial_type: str, temporal_type: str, 
                 process_order: str = 'spatial_first', adj_matrix: torch.Tensor = None):
        super().__init__()
        self.process_order = process_order
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # Processors
        self.spatial_processor = SpatialProcessor(config, spatial_type, adj_matrix)
        self.temporal_processor = TemporalProcessor(config, temporal_type)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.num_features)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, num_nodes, num_features]
        batch_size = x.shape[0]
        
        # Feature projection
        x = self.feature_proj(x)  # [B, T, N, H]
        
        if self.process_order == 'spatial_first':
            # Process each time step's spatial features
            spatial_out = []
            for t in range(self.seq_len):
                spatial_out.append(self.spatial_processor(x[:, t]))
            x = torch.stack(spatial_out, dim=1)
            
            # Process temporal features
            x = x.permute(0, 2, 1, 3)  # [B, N, T, H]
            x = x.reshape(batch_size * self.num_nodes, self.seq_len, -1)
            x = self.temporal_processor(x)  # [B*N, 1, H]
            x = x.view(batch_size, self.num_nodes, -1)  # [B, N, H]
            
        else:  # temporal_first
            # Process temporal features
            x = x.permute(0, 2, 1, 3)  # [B, N, T, H]
            x = x.reshape(batch_size * self.num_nodes, self.seq_len, -1)
            x = self.temporal_processor(x)  # [B*N, 1, H]
            x = x.view(batch_size, self.num_nodes, -1)  # [B, N, H]
            
            # Process spatial features
            x = self.spatial_processor(x)
        
        # Generate predictions
        predictions = self.output_proj(x)
        predictions = predictions.view(batch_size, self.num_nodes, self.pred_len, -1)
        predictions = predictions.permute(0, 2, 1, 3)  # [B, P, N, F]
        
        return predictions

def test_backward():
    config = ModelConfig()
    
    # Create dummy adjacency matrix for GCN
    adj_matrix = torch.rand(config.num_nodes, config.num_nodes)
    
    # Test data
    batch_size = 32
    x = torch.randn(batch_size, config.seq_len, config.num_nodes, config.num_features)
    
    spatial_types = ['mlp', 'cnn', 'gcn']
    temporal_types = ['mlp', 'gru', 'attention']
    process_orders = ['spatial_first', 'temporal_first']
    
    results = []
    
    # Test all combinations
    for sp, tp, order in itertools.product(spatial_types, temporal_types, process_orders):
        try:
            # Initialize model
            model = ModularTrafficPredictor(
                config=config,
                spatial_type=sp,
                temporal_type=tp,
                process_order=order,
                adj_matrix=adj_matrix if sp == 'gcn' else None
            )
            
            # Forward pass
            output = model(x)
            
            # Compute loss and backward
            loss = output.mean()
            loss.backward()
            
            # Check if all parameters have gradients
            has_grad = all(p.grad is not None for p in model.parameters())
            
            results.append({
                'spatial': sp,
                'temporal': tp,
                'order': order,
                'success': True,
                'has_grad': has_grad
            })
            print(f"✓ {sp}-{tp}-{order}: Success")
            
        except Exception as e:
            results.append({
                'spatial': sp,
                'temporal': tp,
                'order': order,
                'success': False,
                'error': str(e)
            })
            print(f"✗ {sp}-{tp}-{order}: Failed - {str(e)}")
    
    return results

if __name__ == "__main__":
    results = test_backward()
    
    # Print summary
    print("\nSummary:")
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    print(f"Successful combinations: {success_count}/{total_count}")
    
    # Print failed combinations if any
    failed = [r for r in results if not r['success']]
    if failed:
        print("\nFailed combinations:")
        for f in failed:
            print(f"  {f['spatial']}-{f['temporal']}-{f['order']}: {f['error']}")