import os

import torch
import torch.nn as nn


# Function to count trainable parameters in the model
def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model (nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoints(model: nn.Module, epoch: int, save_dir: str = None):
    """
    Save a PyTorch model to a file with a filename indicating the epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        epoch (int): The current epoch number.
        save_dir (str): The directory where the model will be saved.

    Returns:
        None
    """
    try:
        # 构建保存文件的文件名，例如：'model_checkpoints_10.pth'
        file_name = f'model_checkpoints_{epoch}.pth'
        file_path = os.path.join(save_dir, file_name)

        torch.save(model.state_dict(), file_path)
        print(f'Model saved to {file_path}')
    except Exception as e:
        print(f'Error saving the model: {str(e)}')
