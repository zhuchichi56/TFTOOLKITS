import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from itertools import product
import json
import os
import matplotlib.pyplot as plt
from config import Config
from dataset import TrafficDataset
from utils import RMSE, MAE, MAPE, plot_predictions
from model import ModularTrafficPredictor
import numpy as np



# def load_from_checkpoint(model,checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     model.load_state_dict(checkpoint["model_state_dict"])
#     return model




class TrafficModelModule(LightningModule):
    def __init__(self, config, spatial, temporal, spatial_first, adj_matrix, std_dev, mean):
        super().__init__()
        self.save_hyperparameters(ignore=[adj_matrix, std_dev, mean])
        self.model = ModularTrafficPredictor(
            config, spatial=spatial, temporal=temporal, spatial_first=spatial_first, adj_matrix=adj_matrix
        )
        self.std_dev = std_dev
        self.mean = mean
        self.loss_fn = nn.MSELoss()
        
        # 添加预测结果和真实值的列表
        self.test_x = []
        self.test_predictions = []
        self.test_targets = []
        self.predictions_cache = None  # 添加缓存属性
        self.targets_cache = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        self.log("val_loss", val_loss, prog_bar=True)

        # 计算指标
        y_pred = y_pred * self.std_dev + self.mean
        y = y * self.std_dev + self.mean
        mae = MAE(y, y_pred)
        rmse = RMSE(y, y_pred)
        mape = MAPE(y, y_pred)

        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        test_loss = self.loss_fn(y_pred, y)
        
        # 反归一化
        x = x * self.std_dev + self.mean
        y_pred_denorm = y_pred * self.std_dev + self.mean
        y_denorm = y * self.std_dev + self.mean
        
        # 存储预测值和真实值
        self.test_x.append(x.cpu().numpy())
        self.test_predictions.append(y_pred_denorm.cpu().numpy())
        self.test_targets.append(y_denorm.cpu().numpy())

        # 计算指标
        # print(f"y_denorm shape: {y_denorm.shape}") # [50, 12, 207, 2])
        # print(f"y_pred_denorm shape: {y_pred_denorm.shape}") #[50, 12, 207, 2])
        mae = MAE(y_denorm, y_pred_denorm)
        rmse = RMSE(y_denorm, y_pred_denorm)
        mape = MAPE(y_denorm, y_pred_denorm)
        
    
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)
        self.log("test_mape", mape, prog_bar=True)
        return {"loss": test_loss}
    
    def on_test_epoch_end(self):
        # 在测试结束时合并所有批次的预测结果
        all_x = np.concatenate(self.test_x, axis=0)
        all_predictions = np.concatenate(self.test_predictions, axis=0)
        all_targets = np.concatenate(self.test_targets, axis=0)
       
    
        # 保存到缓存
        self.x_cache = all_x
        self.predictions_cache = all_predictions
        self.targets_cache = all_targets
        
        # 清空存储的预测结果（为下一次测试做准备）
        self.test_x = []
        self.test_predictions = []
        self.test_targets = []
        
        return {"predictions": all_predictions, "targets": all_targets}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config.learning_rate, weight_decay=self.hparams.config.weight_decay)



def train_model(spatial, temporal, spatial_first, config, test_data=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(config.checkpoint_dir, f"{spatial}-{temporal}-{'spatial_first' if spatial_first else 'temporal_first'}")

    # Load dataset
    dataset = TrafficDataset(config)
    train_sequences, train_targets, val_sequences, val_targets = dataset.get_train_val_data()
    adj_mat, std_dev, mean = dataset.adj_mat, dataset.std_dev, dataset.mean

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_sequences), 
        torch.FloatTensor(train_targets)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.FloatTensor(val_sequences), 
        torch.FloatTensor(val_targets)
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize Lightning module
    model = TrafficModelModule(
        config=config, spatial=spatial, temporal=temporal, spatial_first=spatial_first, 
        adj_matrix=adj_mat, std_dev=std_dev, mean=mean
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, filename="best_model.ckpt", save_top_k=1, monitor="val_loss"
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config.patience, mode="min")

    # Train the model
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # 使用1个GPU或CPU
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=checkpoint_dir
    )
    if not test_data:
        trainer.fit(model, train_loader, val_loader)
        

    # Test the model
    # best_model_path = checkpoint_callback.best_model_path
    best_model_path = os.path.join(checkpoint_dir, "best_model.ckpt")
    logger.info(f"Best model saved at: {best_model_path}")

    # Load the best model for testing
    model = TrafficModelModule.load_from_checkpoint(best_model_path, config=config, spatial=spatial, temporal=temporal, spatial_first=spatial_first, adj_matrix=adj_mat, std_dev=std_dev, mean=mean)
    trainer.test(model, val_loader)
    
    # 从缓存获取预测结果
    x = model.x_cache
    predictions = model.predictions_cache
    targets = model.targets_cache
    logger.info(f"x shape: {x.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    
    plot_predictions(x, predictions, targets, path_dir=checkpoint_dir)
    

    return {
        **trainer.logged_metrics,  # 测试指标
    }





def train_all_combinations(config):
    spatial_types = ['mlp', 'cnn', 'gcn']
    temporal_types = ['mlp', 'gru', 'attention'] 
    results = {}

    # Test all combinations with both spatial_first=True and False
    for spatial_first in [True, False]:
        for spatial, temporal in product(spatial_types, temporal_types):
            model_name = f"{spatial}-{temporal}-{'spatial_first' if spatial_first else 'temporal_first'}"
            logger.info(f"\nTesting combination: {model_name}")
            metrics = train_model(spatial, temporal, spatial_first=spatial_first, config=config, test_data=True)
            
            # 将 Tensor 转换为 Python 原生类型
            results[model_name] = {
                "mae": float(metrics["val_mae"]) if "val_mae" in metrics else None,
                "rmse": float(metrics["val_rmse"]) if "val_rmse" in metrics else None,
                "mape": float(metrics["val_mape"]) if "val_mape" in metrics else None,
            }
            
    return results



if __name__ == '__main__':
    config = Config()
    train_all_combinations(config)



