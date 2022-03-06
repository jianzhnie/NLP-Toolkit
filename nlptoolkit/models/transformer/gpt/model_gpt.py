import logging

import torch
import torch.nn as nn
from torch import Tensor

from nlptoolkit.models.attention.multihead_attn import Attention

logger = logging.getLogger(__name__)


class GPTConfig:
    """base GPT config, params common to all GPT versions."""
    embd_dropout = 0.1
    resid_dropout = 0.1
    attn_dropout = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


def generate_square_subsequent_mask(self, sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.nhead = n_head
        # key, query, value projections for all heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # output projection
        self.output_linear = nn.Linear(d_model, d_model)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size,
                                  block_size)).view(1, 1, block_size,
                                                    block_size))

        self.attention = Attention(dropout=dropout)

        self.n_head = n_head

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            self.mask = mask
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query = self.query_linear(query).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # key = self.key_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # value = self.value_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query = self.query_linear(query).view(nbatches, -1, self.n_head,
                                              self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(nbatches, -1, self.n_head,
                                        self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(nbatches, -1,
                                              self.n_head, self.d_k).transpose(
                                                  1, 2).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self_attn = self.attention(query, key, value, mask=self.mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.n_head * self.d_k)
        # 4) linear proj output
        x = self.output_linear(x)
        output = self.resid_dropout(x)
        return output, self_attn


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(),
        )

    def forward(self, src, src_mask):
        src2 = self.layer_norm1(src)
        src2 = self.attn(src2, src2, src2, mask=src_mask)[0]
        src = src + src2

        src2 = self.layer_norm2(src)
        src2 = self.mlp(src2)
        src = src + src2
        return src
