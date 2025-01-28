import os
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod
from config import *

__all__ = ['TrafficDataset']

# # 数据集说明 PeMSD7Dataset: 

# ## 1. 数据集背景
# PeMSD7 数据集来源于 **California Department of Transportation (Caltrans)** 提供的 **Performance Measurement System (PeMS)**。它记录了来自加州公路传感器的交通数据，包括流量、速度和占用率等信息。**PeMSD7** 是其中一个子集，专门选取了特定区域的交通流数据。

# ---

# ## 2. 数据集组成

# ### 交通流量数据 (`PeMSD7_V_228.csv`)
# - 包含 **228 个传感器**在一段时间内记录的交通流量数据。
# - 数据格式为一个二维矩阵：
#   - **行**：表示时间步（一般为每 5 分钟一个采样点）。
#   - **列**：表示不同位置（传感器）的流量值。
# - **数据特点**：
#   - **时间序列**：每个传感器按时间记录流量数据。
#   - **空间依赖性**：相邻或相近的传感器流量具有一定的空间相关性。

# ### 距离矩阵 (`PeMSD7_W_228.csv`)
# - 记录了传感器之间的物理距离（通常为欧几里得距离或道路距离）。
# - 距离矩阵是一个对称的二维矩阵，矩阵元素 \( w_{ij} \) 表示传感器 \( i \) 和 \( j \) 的距离。

# ---

# ## 3. 数据处理

# ### (1) 节点特征 (`node_values`)
# - 每个传感器被视为图中的一个节点。
# - 节点特征表示的是时间序列流量数据。
# - 数据处理步骤：
#   1. **归一化**：对流量值进行标准化，减去均值并除以标准差。
#   2. **重塑格式**：数据形状转换为 \( (T, N, F) \)，其中：
#      - \( T \)：时间步数。
#      - \( N \)：节点数（传感器数）。
#      - \( F \)：特征维度（此处为 1，即流量值）。

# ### (2) 邻接矩阵 (`adj_mat`)
# - 描述传感器之间的连接关系，基于距离矩阵计算得出。
# - 邻接矩阵权重计算公式：
#   \[
#   w_{ij} = \exp\left(-\frac{d_{ij}^2}{\sigma}\right)
#   \]
# - **计算细节**：
#   - **高斯核函数**：根据传感器之间的距离 \( d_{ij} \)，分配权重。越近的传感器权重越高。
#   - **阈值过滤**：设置权重阈值，过滤掉权重较低的边（表示距离过远的节点间无连接）。
#   - **二值化（可选）**：若 `binary_weights=True`，则将所有非零权重置为 1。


class TDataset(ABC):
    """Abstract base class for traffic datasets"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.node_values = None
        self.adj_mat = None
        self.std_dev = None
        self.mean = None
        self.load_and_process()
    
    @abstractmethod
    def load_and_process(self):
        """Load and process the dataset"""
        pass
    
    def get_train_val_data(self, train_ratio=0.8):
        """Get training and validation sequences"""
        sequences, targets = self._preprocess_data(
            self.node_values, 
            self.config.seq_len, 
            self.config.pred_len
        )
        return self._split_data(sequences, targets, train_ratio)

    def _preprocess_data(self, data, seq_len, pred_len):
        """Create sequences from data, 这个就是滑动窗口, 每次滑动一个格子"""
        sequences, targets = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            sequences.append(data[i:(i + seq_len)])
            targets.append(data[(i + seq_len):(i + seq_len + pred_len)])
        sequences = np.array(sequences)
        targets = np.array(targets)
        logger.info(f"Preprocessed sequences shape: {sequences.shape}, targets shape: {targets.shape}")
        return sequences, targets
    
    def _split_data(self, sequences, targets, train_ratio=0.8):
        """Split data into training and validation sets"""
        n = len(sequences)
        train_size = int(n * train_ratio)
        
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        val_sequences = sequences[train_size:]
        val_targets = targets[train_size:]
        
        return train_sequences, train_targets, val_sequences, val_targets

class MetrLADataset(TDataset):
    """Dataset for loading METR-LA traffic data"""
    
    def load_and_process(self):
        """Load and return node values and adjacency matrix"""
        self.adj_mat = np.load(self.config.distance_matrix_path)
        self.node_values = np.load(self.config.source_path)
        self.std_dev = np.std(self.node_values)
        self.mean = np.mean(self.node_values)   
        self.node_values = (self.node_values - self.mean) / self.std_dev
        
        
        logger.info(f"Loaded adj_mat shape: {self.adj_mat.shape}, node_values shape: {self.node_values.shape}")
        return self.node_values, self.adj_mat
    
    
    def get_train_val_data(self, train_ratio=0.8):
        """Get training and validation sequences"""
        sequences, targets = self._preprocess_data(
            self.node_values, 
            self.config.seq_len, 
            self.config.pred_len
        )
        return self._split_data(sequences, targets, train_ratio)



class PeMSD7Dataset(TDataset):
    """Dataset for loading PeMSD7 traffic data"""
    
    def load_and_process(self):
        """Load and process PeMSD7 data"""
        # Read distance matrix and convert to weights
        distances = pd.read_csv(self.config.distance_matrix_path, header=None).values
        self.adj_mat = self.distance_to_weight(
            distances,
            sigma=self.config.sigma,
            threshold=self.config.threshold,
            binary_weights=self.config.binary_weights
        )
        
        # Process node values
        data = pd.read_csv(self.config.source_path, header=None).values
        self.std_dev = np.std(data)
        self.mean = np.mean(data)
        data = (data - self.mean) / self.std_dev
        T, N = data.shape
        self.node_values = data.reshape(T, N, 1)
        logger.info(f"Processed shapes - node_values: {self.node_values.shape}, adj_mat: {self.adj_mat.shape}")
        
        return self.node_values, self.adj_mat
    
    def get_train_val_data(self, train_ratio=0.8):
        """Get training and validation sequences"""
        sequences, targets = self._preprocess_data(
            self.node_values, 
            self.config.seq_len, 
            self.config.pred_len
        )
        return self._split_data(sequences, targets, train_ratio)    
    

    @staticmethod
    def distance_to_weight(distances, sigma, threshold, binary_weights):
        """Convert distances between nodes into weight matrix"""
        n = distances.shape[0]
        distances = distances / 1000.0  # Convert to kilometers
        squared_dist = distances * distances
        mask = np.ones([n, n]) - np.identity(n)
        
        weights = np.exp(-squared_dist / sigma) * (np.exp(-squared_dist / sigma) >= threshold) * mask

        if binary_weights:
            weights[weights > 0] = 1
            weights += np.identity(n)

        return weights



class TrafficDataset:
    def __init__(self, config):
        if config.dataset_name.lower() == "pemsd7":
            dataset_class = PeMSD7Dataset
        elif config.dataset_name.lower() == "metr-la": 
            dataset_class = MetrLADataset
        else:
            raise ValueError(f"Dataset {config.dataset_name} not supported")
        self.dataset = dataset_class(config)
        logger.info(f"Initialized {config.dataset_name} dataset")
    
    def get_data_loaders(self, batch_size, train_ratio=0.8):
        """Get DataLoader objects for training and validation"""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        train_sequences, train_targets, val_sequences, val_targets = self.get_train_val_data(train_ratio)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_targets)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_targets)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info(f"Created data loaders - train: {len(train_loader)} batches, val: {len(val_loader)} batches")
        return train_loader, val_loader

    def get_train_val_data(self, train_ratio=0.8):
        """Delegate method to get training and validation data"""
        return self.dataset.get_train_val_data(train_ratio)
    
    @property
    def node_values(self):
        """Get node values from the underlying dataset"""
        return self.dataset.node_values
    
    @property
    def adj_mat(self):
        """Get adjacency matrix from the underlying dataset"""
        return self.dataset.adj_mat
    
    @property
    def std_dev(self):
        """Get standard deviation from the underlying dataset"""
        return self.dataset.std_dev
    
    @property
    def mean(self):
        """Get mean from the underlying dataset"""
        return self.dataset.mean



if __name__ == "__main__":
    from config import Config
    import torch
    
    # Test basic functionality
    dataset = TrafficDataset(Config(dataset_name="PeMSD7"))
    
    # Test getting raw data
    train_sequences, train_targets, val_sequences, val_targets = dataset.get_train_val_data()
    print("Raw data shapes:", train_sequences.shape, train_targets.shape)
    
    # Test getting data loaders
    train_loader, val_loader = dataset.get_data_loaders(batch_size=32)
    print("Number of batches:", len(train_loader), len(val_loader))
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        print("Sample batch shapes:", batch_x.shape, batch_y.shape)
        break

