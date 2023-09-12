import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    """Compute 'Scaled Dot Product Attention'."""
    def __init__(self, dropout, **kwargs):
        super(Attention, self).__init__(**kwargs)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # mask 步骤, 用 -1e9 代表 负无穷
        # 在计算 得分的时候, 负无穷那部分可以忽略
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        output = torch.matmul(p_attn, value)

        return output, p_attn


class MultiHeadAttention(nn.Module):
    """In the case of Encoder, Q,K,V, will simply the identical copies of the
    embedding vector.

        They will have the same dimensions: Batch * Seq_len * d_model.

    In multi-head attention we split the embedding vector into N heads,
        so they will have the dimensions: Batch * Seq_len * nhead * (d_model/nhead).

    step1: Given query, key, value, split into n_heads
    step2: Calculate attention using the resulting Q/K/V Matrix
    step3: Concate the results
    step4: Multiply with weight matrix Wo to produce the output of the layer
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        """Take in model size and number of heads."""

        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        # self.query_linear = nn.Linear(d_model, d_model)
        # self.key_linear = nn.Linear(d_model, d_model)
        # self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query = self.query_linear(query).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # key = self.key_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # value = self.value_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)

        query, key, value = [
            linear(x).view(nbatches, -1, self.nhead, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self_attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.nhead * self.d_k)
        # 4) linear proj output
        output = self.output_linear(x)
        return output, self_attn
