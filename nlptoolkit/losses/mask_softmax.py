'''
Author: jianzhnie
Date: 2021-12-20 15:43:53
LastEditTime: 2021-12-22 09:33:07
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn as nn


class MaskedSoftmaxCELoss(nn.Module):
    """
    The softmax cross-entropy loss with masks.

    This loss is suitable for sequence-to-sequence models where sequences have varying lengths.
    It applies a mask to ignore padding elements when computing the loss.

    Args:
        ignore_index (int, optional): Index to ignore when computing the loss. Defaults to -100.

    Attributes:
        ignore_index (int): Index to ignore when computing the loss.
        criterion (nn.CrossEntropyLoss): Cross-entropy loss criterion with predefined settings.

    """
    def __init__(self, ignore_index: int = -100):
        super(MaskedSoftmaxCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(reduction='none',
                                             ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, label: torch.Tensor,
                valid_len: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked softmax cross-entropy loss.

        Args:
            pred (torch.Tensor): Predicted scores (logits) of shape (batch_size, seq_len, vocab_size).
            label (torch.Tensor): Ground truth labels of shape (batch_size, seq_len).
            valid_len (torch.Tensor): Lengths of valid elements in each sequence of shape (batch_size,).

        Returns:
            weighted_loss (torch.Tensor): Weighted loss of shape (batch_size,).

        """
        mask = self.generate_mask(label, valid_len)
        loss = self.criterion(pred.permute(0, 2, 1), label)
        weighted_loss = torch.sum(loss * mask) / torch.sum(mask)
        return weighted_loss

    def generate_mask(self, label: torch.Tensor,
                      valid_len: torch.Tensor) -> torch.Tensor:
        """
        Generate a mask to ignore padding elements.

        Args:
            label (torch.Tensor): Ground truth labels of shape (batch_size, seq_len).
            valid_len (torch.Tensor): Lengths of valid elements in each sequence of shape (batch_size,).

        Returns:
            mask (torch.Tensor): Mask of shape (batch_size, seq_len) where padding elements are set to 0.

        """
        batch_size, seq_len = label.size()
        mask = torch.arange(seq_len,
                            dtype=torch.float32, device=label.device).expand(
                                batch_size, seq_len) < valid_len.unsqueeze(1)
        return mask.float()
