from dataclasses import dataclass, fields

# Model Configuration
@dataclass
class ModelConfig:
    num_nodes: int = 207  
    hidden_dim: int = 256

# Training Configuration
@dataclass
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 50    
    weight_decay: float = 5e-5
    learning_rate: float = 3e-4
    checkpoint_dir: str = "./runs"
    project_name: str = "stgat"
    eval_interval: int = 10
    patience: int = 10  

# Dataset Specific Configurations
@dataclass
class PeMSD7Config:
    """Configuration class for PeMSD7 dataset"""
    distance_matrix_path: str = './data/PeMSD7/PeMSD7_W_228.csv'
    source_path: str = './data/PeMSD7/PeMSD7_V_228.csv'
    binary_weights: bool = True
    sigma: float = 0.1
    threshold: float = 0.5
    seq_len: int = 12
    pred_len: int = 9
    num_features: int = 1

@dataclass
class METRLAConfig:
    """Configuration class for METR-LA dataset"""
    distance_matrix_path: str = './data/METR-LA/adj_mat.npy'
    source_path: str = './data/METR-LA/node_values.npy'
    seq_len: int = 12
    pred_len: int = 12
    num_features: int = 2

# Main Data Configuration
@dataclass
class DataConfig:
    dataset_name: str = "METR-LA"  # or "METR-LA" or PeMSD7
    
    def __post_init__(self):
        # Dynamically load the dataset-specific config
        config_classes = {
            "PeMSD7": PeMSD7Config,
            "METR-LA": METRLAConfig,
        }
        if self.dataset_name not in config_classes:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        
        self.dataset_config = config_classes[self.dataset_name]()
        for field in fields(self.dataset_config):
            setattr(self, field.name, getattr(self.dataset_config, field.name))



# Main Configuration Class
@dataclass
class Config(ModelConfig, TrainingConfig, DataConfig):
    
    
    pass
        

# 测试能否直接访问 seq_len 和 pred_len
if __name__ == "__main__":
    config = Config(dataset_name="METR-LA")
    print(config.seq_len)  # 应该输出 12
    print(config.pred_len)  # 应该输出 12