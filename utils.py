import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# traffic flow 的维度是 [batch_size, n_slot, n_node, n_feature] 
# print(f"y_denorm shape: {y_denorm.shape}") # [50, 12, 207, 2])
# print(f"y_pred_denorm shape: {y_pred_denorm.shape}") #[50, 12, 207, 2])
        
def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth
    :param v_: torch array, prediction
    :return: torch scaler, RMSE averages on all elements of input
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2))

def MAE(v, v_):
    """
    Mean Absolute Error
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


# #  test_mape               273967968.0
# def MAPE(v, v_):
#     """
#     Mean absolute percentage error
#     :param v: torch array, ground truth (should not contain zeros)
#     :param v_: torch array, prediction
#     :return: torch scalar, MAPE averages on all elements of input
#     """
#     v = v[:, :, :, 0]
#     v_ = v_[:, :, :, 0]
#     mask = v != 0
#     return torch.mean(torch.abs((v_ - v)[mask] / v[mask]) * 100)

# test_mae             16.75347900390625
# test_mape               138026288.0
# test_rmse            4.266353607177734


# 2025-01-28 09:01:28.285 | INFO     | __main__:train_model:177 - Predictions shape: (2531, 9, 228, 1)
# 2025-01-28 09:01:28.285 | INFO     | __main__:train_model:178 - Targets shape: (2531, 9, 228, 1) # batch_size, n_slot, n_node, n_feature

def plot_predictions(x,predictions, targets, node_idx=0, path_dir=None):
    """Plot predictions vs ground truth for a specific node.
    
    Args:
        predictions: numpy array of shape (batch_size, n_slot, n_node, n_feature)
        targets: numpy array of shape (batch_size, n_slot, n_node, n_feature) 
        node_idx: which node to plot predictions for
    """
    # Get predictions and targets for specified node
    node_x = x[:, :, node_idx, 0]  # (batch_size, n_slot)
    node_preds = predictions[:, :, node_idx, 0]  # (batch_size, n_slot)
    node_targets = targets[:, :, node_idx, 0]
    
    # Plot first batch
    plt.figure(figsize=(10, 6))
    # Plot history
    history_len = node_x.shape[1]
    time_x = range(history_len)
    plt.plot(time_x, node_x[0], label='History')
    
    # Plot predictions and ground truth
    pred_len = node_preds.shape[1]
    time_pred = range(history_len, history_len + pred_len)
    plt.plot(time_pred, node_preds[0], label='Predictions')
    plt.plot(time_pred, node_targets[0], label='Ground Truth')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Flow')
    plt.title(f'Traffic Flow Predictions vs Ground Truth for Node {node_idx}')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(path_dir, f"traffic_predictions_node_{node_idx}.png"))
    plt.close()


# def plot_predictions(y_pred, y_truth, node, config):
#     s = y_truth.shape  # (27, 11400, 9) for test_data
#     y_truth = y_truth.reshape(
#         s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1]  # (27, 50, 228, 9)
#     )
#     # just get the first prediction out for the nth node
#     y_truth = y_truth[:, :, node, 0]  # (27, 50)
#     # Flatten to get the predictions for entire test dataset
#     y_truth = torch.flatten(y_truth)  # (1350)
#     day0_truth = y_truth[: config["N_SLOT"]]  # (268)

#     # Calculate the predicted
#     s = y_pred.shape
#     y_pred = y_pred.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
#     # just get the first prediction out for the nth node
#     y_pred = y_pred[:, :, node, 0]
#     # Flatten to get the predictions for entire test dataset
#     y_pred = torch.flatten(y_pred)
#     # Just grab the first day
#     day0_pred = y_pred[: config["N_SLOT"]]

#     t = [t for t in range(0, config["N_SLOT"])]  # 268

#     plt.plot(t, day0_pred, label="ST-GAT")
#     plt.plot(t, day0_truth, label="truth")
#     plt.xlabel("Time (minutes)")
#     plt.ylabel("Speed prediction")
#     plt.title("Predictions of traffic over time")
#     plt.legend()

#     file_path = f"./assets/traffic_on_node{node}_day0.png"
#     assets_dir = os.path.dirname(file_path)
#     if not os.path.exists(assets_dir):
#         os.makedirs(assets_dir)
#     plt.savefig(file_path)
#     plt.show()
    
    
    
if __name__ == "__main__":
    # test plot_predictions
    # predictions = np.random.rand(2531, 9, 228, 1)
    # targets = np.random.rand(2531, 9, 228, 1)
    # plot_predictions(predictions, targets, node_idx=0)
    
    # test mae:
    y_denorm = np.random.rand(2531, 9, 228, 1)
    y_pred_denorm = np.random.rand(2531, 9, 228, 1)
    # tensor
    y_denorm = torch.tensor(y_denorm)
    y_pred_denorm = torch.tensor(y_pred_denorm)
    mae = MAE(y_denorm, y_pred_denorm)

    print(f"mae: {mae}")