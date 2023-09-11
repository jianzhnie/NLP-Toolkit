'''
Author: jianzhnie
Date: 2021-12-20 15:43:53
LastEditTime: 2021-12-22 09:33:07
LastEditors: jianzhnie
Description:

'''

from typing import List, Union

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
    def __init__(self):
        super(MaskedSoftmaxCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

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

    def generate_mask(
            self, inputs: torch.Tensor,
            valid_len: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Generate a mask to ignore padding elements.

        Args:
            inputs (torch.Tensor): Ground truth inputs of shape (batch_size, seq_len).
            valid_len (torch.Tensor): Lengths of valid elements in each sequence of shape (batch_size,).

        Returns:
            mask (torch.Tensor): Mask of shape (batch_size, seq_len) where padding elements are set to 0.

        """
        batch_size, seq_len = inputs.size()
        mask = torch.arange(seq_len,
                            dtype=torch.float32, device=inputs.device).expand(
                                batch_size, seq_len) < valid_len.unsqueeze(1)
        return mask

    def sequence_mask(self,
                      inputs: torch.Tensor,
                      valid_len: Union[List[int], torch.Tensor],
                      value: float = 0.0) -> torch.Tensor:
        """
        Apply a sequence mask to the input tensor to mask out elements beyond the valid lengths of each sequence.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, ...).
            valid_len (Union[List[int], torch.Tensor]): List or tensor of valid sequence lengths for each batch element.
            value (float, optional): Value to fill masked elements with (default is 0.0).

        Returns:
            torch.Tensor: Masked tensor with the same shape as the input tensor.

        Example:
            >>> inputs = torch.tensor([[1, 2, 3, 4, 5],
            ...                   [6, 7, 8, 9, 10],
            ...                   [11, 12, 13, 14, 15]], dtype=torch.float32)
            >>> valid_len = [2, 3, 4]
            >>> masked = sequence_mask(inputs, valid_len)
            >>> print(masked)
            tensor([[1., 2., 0., 0., 0.],
                    [6., 7., 8., 0., 0.],
                    [11., 12., 13., 14., 0.]])
        """
        # Get the maximum sequence length
        maxlen = inputs.size(1)
        # Create a mask for each sequence
        mask = torch.arange(maxlen, dtype=torch.float32,
                            device=inputs.device)[None, :] < valid_len[:, None]
        # Apply the mask to the input tensor
        inputs[~mask] = value
        return inputs


if __name__ == '__main__':
    Loss = MaskedSoftmaxCELoss()
    inputs = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=torch.float32)
    valid_len = torch.tensor([2, 3, 4])
    masked = Loss.generate_mask(inputs, valid_len)
    print(masked)
