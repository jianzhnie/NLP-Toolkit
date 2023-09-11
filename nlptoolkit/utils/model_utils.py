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
