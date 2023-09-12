import os
from typing import Union

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


def clip_gradients(model: Union[nn.Module, object], theta: float):
    """
    Clip gradients to prevent exploding gradients during training.

    Args:
        model (nn.Module or object): The neural modelwork model or object containing parameters.
        theta (float): The threshold value for gradient clipping.

    Returns:
        None

    ## Example usage:
    ```python
    clip_gradients(your_model, 1.0)  # Clip gradients with a threshold of 1.0
    ```
    """
    if isinstance(model, nn.Module):
        # If `model` is an instance of nn.Module (a PyTorch model)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        # If `model` is a custom object with gradient-containing parameters
        params = model.params

    # Calculate the L2 norm of the gradients
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))

    # Clip gradients if their norm exceeds the threshold
    if norm > theta:
        for param in params:
            param.grad *= theta / norm
