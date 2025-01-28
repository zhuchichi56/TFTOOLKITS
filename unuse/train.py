class SimpleTrafficPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # 1. 减小hidden_dim,避免特征空间过大
        mid_dim = config.hidden_dim // 2
        
        # 2. 特征投影加入LayerNorm
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU()
        )
    
        
        # 3. Temporal MLP - 渐进式压缩
        self.temporal_mlp = nn.Sequential(
            nn.Linear(config.seq_len * mid_dim, config.seq_len * mid_dim // 2),
            nn.LayerNorm(config.seq_len * mid_dim // 2),
            nn.ReLU(),
            nn.Linear(config.seq_len * mid_dim // 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 4. Spatial MLP - 加入残差连接
        self.spatial_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 5. 输出层渐进式扩张
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.pred_len * config.num_features)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 特征投影
        x = self.feature_proj(x)
        
        # 处理时序
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * self.num_nodes, -1)
        temporal_features = self.temporal_mlp(x)
        temporal_features = temporal_features.view(batch_size, self.num_nodes, -1)
        
        # 空间特征处理 + 残差
        spatial_features = self.spatial_proj(temporal_features)
        spatial_features = spatial_features + temporal_features
        
        # 生成预测
        predictions = self.output_proj(spatial_features)
        predictions = predictions.view(batch_size, self.num_nodes, self.pred_len, self.num_features)
        predictions = predictions.permute(0, 2, 1, 3)
        
        return predictions

# Spatial
class TrafficPredictor_CNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # 1. 减小hidden_dim，避免特征空间过大
        mid_dim = config.hidden_dim // 2
        
        # 2. 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU()
        )
        
        # 3. 空间CNN模块
        self.spatial_conv = nn.Sequential(
            # 使用1x1卷积调整通道数
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1),
            nn.LayerNorm([mid_dim, self.num_nodes, self.seq_len]),
            nn.ReLU(),
            # 3x3卷积捕捉局部空间关系
            nn.Conv2d(mid_dim, mid_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.LayerNorm([mid_dim, self.num_nodes, self.seq_len]),
            nn.ReLU(),
            # 1x1卷积整合特征
            nn.Conv2d(mid_dim, config.hidden_dim, kernel_size=1),
            nn.LayerNorm([config.hidden_dim, self.num_nodes, self.seq_len]),
            nn.ReLU()
        )
        
        # 4. 时序MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(config.seq_len, config.seq_len // 2),
            nn.LayerNorm(config.seq_len // 2),
            nn.ReLU(),
            nn.Linear(config.seq_len // 2, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.ReLU()
        )
        
        # 5. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_features)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, num_nodes, num_features]
        batch_size = x.shape[0]
        
        # 1. 特征投影
        x = self.feature_proj(x)  # [batch_size, seq_len, num_nodes, mid_dim]
        
        # 2. 重排序为CNN输入格式
        x = x.permute(0, 3, 2, 1)  # [batch_size, mid_dim, num_nodes, seq_len]
        
        # 3. 空间CNN处理
        spatial_features = self.spatial_conv(x)  # [batch_size, hidden_dim, num_nodes, seq_len]
        
        # 4. 时序处理
        # 重排序以处理每个节点的时序
        temporal_features = spatial_features.permute(0, 2, 1, 3)  # [batch_size, num_nodes, hidden_dim, seq_len]
        temporal_features = self.temporal_mlp(temporal_features)  # [batch_size, num_nodes, hidden_dim, pred_len]
        
        # 5. 输出处理
        output = temporal_features.permute(0, 3, 1, 2)  # [batch_size, pred_len, num_nodes, hidden_dim]
        predictions = self.output_proj(output)  # [batch_size, pred_len, num_nodes, num_features]
        
        return predictions

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        output = torch.matmul(adj, support)     # [batch_size, num_nodes, out_features]
        return output + self.bias

class TrafficPredictor_GCN(nn.Module):
    def __init__(self, config: ModelConfig, adj_matrix: torch.Tensor):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # 处理邻接矩阵
        adj_matrix = torch.FloatTensor(adj_matrix)
        # 添加自环
        adj_matrix = adj_matrix + torch.eye(self.num_nodes)
        # 计算度矩阵
        D = torch.sum(adj_matrix, dim=1)
        D = torch.diag(1.0 / torch.sqrt(D))
        # 标准化邻接矩阵
        self.register_buffer('adj', torch.matmul(torch.matmul(D, adj_matrix), D))
        
        # 1. 减小hidden_dim，避免特征空间过大
        mid_dim = config.hidden_dim // 2
        
        # 2. 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU()
        )
        
        # 3. 空间GCN模块
        self.spatial_gcn = nn.ModuleList([
            nn.Sequential(
                GraphConvolution(mid_dim, mid_dim),
                nn.LayerNorm([self.num_nodes, mid_dim]),
                nn.ReLU()
            ),
            nn.Sequential(
                GraphConvolution(mid_dim, config.hidden_dim),
                nn.LayerNorm([self.num_nodes, config.hidden_dim]),
                nn.ReLU()
            )
        ])
        
        # 4. 时序MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(config.seq_len, config.seq_len // 2),
            nn.LayerNorm(config.seq_len // 2),
            nn.ReLU(),
            nn.Linear(config.seq_len // 2, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.ReLU()
        )
        
        # 5. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_features)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, num_nodes, num_features]
        batch_size = x.shape[0]
        
        # 1. 特征投影
        x = self.feature_proj(x)  # [batch_size, seq_len, num_nodes, mid_dim]
        
        # 2. 重排序并处理每个时间步的图卷积
        spatial_features = []
        for t in range(self.seq_len):
            # 获取当前时间步的特征
            curr_feat = x[:, t]  # [batch_size, num_nodes, mid_dim]
            
            # 应用GCN层
            for gcn_layer in self.spatial_gcn:
                curr_feat = gcn_layer[0](curr_feat, self.adj)  # GraphConv
                curr_feat = gcn_layer[1:](curr_feat)          # LayerNorm + ReLU
            
            spatial_features.append(curr_feat)
        
        # 3. 堆叠所有时间步的结果
        spatial_features = torch.stack(spatial_features, dim=1)  
        # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # 4. 时序处理
        # 重排序以处理每个节点的时序
        temporal_features = spatial_features.permute(0, 2, 3, 1)  
        # [batch_size, num_nodes, hidden_dim, seq_len]
        temporal_features = self.temporal_mlp(temporal_features)  
        # [batch_size, num_nodes, hidden_dim, pred_len]
        
        # 5. 输出处理
        output = temporal_features.permute(0, 3, 1, 2)  
        # [batch_size, pred_len, num_nodes, hidden_dim]
        predictions = self.output_proj(output)  
        # [batch_size, pred_len, num_nodes, num_features]
        
        return predictions

# temporal
class SimpleTrafficPredictor_GRU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # 1. 减小hidden_dim
        mid_dim = config.hidden_dim // 2
        
        # 2. 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU()
        )
        
        # 3. Temporal GRU
        self.temporal_gru = nn.GRU(
            input_size=mid_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True  # 使用双向GRU
        )
        
        # GRU输出的特征维度将是hidden_dim*2(因为双向)
        self.gru_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 4. Spatial MLP (保持不变)
        self.spatial_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 5. 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.pred_len * config.num_features)
        )
        
    def forward(self, x):
        # [B, T, N, F] -> [B, T, N, D]
        x = self.feature_proj(x)
        
        # [B, T, N, D] -> [B, N, T, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.seq_len, x.size(-1))
        
        # [B*N, T, D] -> [B*N, T, 2D]
        gru_out, _ = self.temporal_gru(x)
        
        # [B*N, T, 2D] -> [B*N, 2D] -> [B*N, D] -> [B, N, D]
        temporal_features = gru_out[:, -1]
        temporal_features = self.gru_proj(temporal_features)
        temporal_features = temporal_features.view(-1, self.num_nodes, temporal_features.size(-1))
        
        # [B, N, D] -> [B, N, D]
        spatial_features = self.spatial_proj(temporal_features)
        spatial_features = spatial_features + temporal_features
        
        # [B, N, D] -> [B, N, P*F]: 将特征维度D映射到预测长度P和特征数F的乘积
        # [B, N, P*F] -> [B, N, P, F]: 重塑张量,分离预测长度和特征维度
        # [B, P, N, F]: 调整维度顺序,使预测长度P在第二维,符合模型输出格式 [batch, pred_len, nodes, features]
        predictions = self.output_proj(spatial_features)
        predictions = predictions.view(-1, self.num_nodes, self.pred_len, self.num_features)
        predictions = predictions.permute(0, 2, 1, 3)
        
        return predictions

class SimpleTrafficPredictor_Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_features = config.num_features
        
        # 1. 减小hidden_dim
        mid_dim = config.hidden_dim // 2
        
        # 2. 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.num_features, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU()
        )
        
        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_dim,
            nhead=8,
            dim_feedforward=mid_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )
        
        # 添加最终投影层，将mid_dim转换为hidden_dim
        self.final_proj = nn.Sequential(
            nn.Linear(mid_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 4. Spatial MLP (保持不变)
        self.spatial_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 5. 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.pred_len * config.num_features)
        )
        
    def forward(self, x):
    
        batch_size = x.shape[0]
        x = self.feature_proj(x)
    
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * self.num_nodes, self.seq_len, -1)
        temporal_features = self.transformer_encoder(x)
        temporal_features = temporal_features[:, -1]
        temporal_features = self.final_proj(temporal_features)
        temporal_features = temporal_features.view(batch_size, self.num_nodes, -1)
        
        # 4. 空间特征处理 + 残差
        spatial_features = self.spatial_proj(temporal_features)
        spatial_features = spatial_features + temporal_features
        
        # 5. 生成预测
        predictions = self.output_proj(spatial_features)
        predictions = predictions.view(batch_size, self.num_nodes, self.pred_len, self.num_features)
        predictions = predictions.permute(0, 2, 1, 3)
        return predictions





