import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module: nn.Module, N: int):
    """
    Create N identical copies of the given module.

    Args:
        module (nn.Module): Module to be copied.
        N (int): Number of copies.

    Returns:
        nn.ModuleList: List of copied modules.

    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'.

    Args:
        dropout (float): Dropout probability.

    Attributes:
        dropout (nn.Dropout): Dropout layer (if dropout > 0).

    """
    def __init__(self, dropout: float):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Scaled Dot Product Attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor for attention masking.

        Returns:
            torch.Tensor: Output of the attention.

        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        output = torch.matmul(p_attn, value)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer.

    In multi-head attention we split the embedding vector into N heads,
        so they will have the dimensions: Batch * Seq_len * nhead * (d_model/nhead).

    - step1: Given query, key, value, split into n_heads
    - step2: Calculate attention using the resulting Q/K/V Matrix
    - step3: Concate the results
    - step4: Multiply with weight matrix Wo to produce the output of the layer

    Args:
        d_model (int): Model size.
        nhead (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model) ->  (batch_size, seq_len, self.nhead, self.d_k)
        new_x_shape = x.size()[:-1] + (self.nhead, self.d_k)
        x = x.view(new_x_shape)
        #  (batch_size, self.nhead, seq_len, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor. (batch_size, seq_len, d_model)
            key (torch.Tensor): Key tensor. (batch_size, seq_len, d_model)
            value (torch.Tensor): Value tensor. (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask tensor for attention masking.

        Returns:
            torch.Tensor: Output of the attention.

        """
        # Same mask applied to all h heads.
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => nhead * d_k
        mixed_query_layer = self.query_linear(query)
        mixed_key_layer = self.key_linear(key)
        mixed_value_layer = self.value_linear(value)

        # Transpose the projections into multi-head
        # (batch_size, seq_len, d_model) => (batch_size, nhead, seq_len, d_k)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query_layer, key_layer, value_layer, mask=mask)
        # x shape: (batch_size, nhead, seq_len, d_k)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.nhead * self.d_k)
        # x shape: (batch_size, seq_len, nhead * d_k)

        # 4) linear proj output
        output = self.output_linear(x)
        # output shape: (batch_size, seq_len, d_model)
        return output
