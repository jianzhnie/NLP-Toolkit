'''
Author: jianzhnie
Date: 2021-12-16 14:18:44
LastEditTime: 2022-03-24 10:45:03
LastEditors: jianzhnie
Description:

'''

import sys

import torch
import torch.nn as nn

sys.path.append('../../../../')
import math
from typing import Optional

from torch import Tensor

from nlptoolkit.losses.mask_softmax import masked_softmax


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状."""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作."""
    # 输入X的形状: (batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    # 输出X的形状: (batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # 输出的形状: (batch_size，查询或者“键－值”对的个数，num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 输出的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


class AdditiveAttention(nn.Module):
    """
    Additive Attention mechanism.

    Args:
        key_size (int): Size of the key vectors.
        query_size (int): Size of the query vectors.
        num_hiddens (int): Number of hidden units in the attention mechanism.
        dropout (float): Dropout probability for regularization.


    Methods:
        forward(queries, keys, values, valid_lens):
            Perform additive attention and return the attention-weighted values.

    """
    def __init__(self, key_size: int, query_size: int, num_hiddens: int,
                 dropout: float):
        """
        Initialize the AdditiveAttention module.

        Args:
            key_size (int): Size of the key vectors.
            query_size (int): Size of the query vectors.
            num_hiddens (int): Number of hidden units in the attention mechanism.
            dropout (float): Dropout probability for regularization.

        """
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Optional[Tensor]) -> Tensor:
        """
        Compute additive attention.

        Args:
            queries (Tensor): The query tensor of shape (batch_size, num_queries, query_size).
            keys (Tensor): The key tensor of shape (batch_size, num_key_value_pairs, key_size).
            values (Tensor): The value tensor of shape (batch_size, num_key_value_pairs, value_dimension).
            valid_lens (Optional[Tensor]): An optional tensor of shape (batch_size,) or (batch_size, num_queries).

        Returns:
            Tensor: The attention-weighted output tensor.

        """
        queries, keys = self.W_q(queries), self.W_k(keys)

        # Broadcast the queries and keys to calculate the attention scores
        # features shape: (batch_size, num_queries, num_key_value_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # Calculate attention scores and apply masking if valid_lens is provided
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = self.w_v(features).squeeze(-1)
        # Calculate the attention-weighted values
        scores = masked_softmax(scores, valid_lens)
        # Apply dropout
        output = self.dropout(scores)
        # output shape: (batch_size, num_queries, value_dimension)
        output = torch.bmm(scores, values)
        return output


class DotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention.

    Args:
        dropout (float): Dropout probability for regularization.

    Methods:
        forward(queries, keys, values, valid_lens=None):
            Perform scaled dot product attention and return the attention-weighted values.

    """
    def __init__(self, dropout: float):
        """
        Initialize the DotProductAttention module.

        Args:
            dropout (float): Dropout probability for regularization.

        """
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                valid_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scaled dot product attention.

        Args:
            queries (torch.Tensor): The query tensor of shape (batch_size, num_queries, d).
            keys (torch.Tensor): The key tensor of shape (batch_size, num_key_value_pairs, d).
            values (torch.Tensor): The value tensor of shape (batch_size, num_key_value_pairs, value_dimension).
            valid_lens (Optional[torch.Tensor]): An optional tensor of shape (batch_size,) or (batch_size, num_queries).

        Returns:
            Tensor: The attention-weighted output tensor.

        """
        d = queries.shape[-1]

        # Compute attention scores using dot product
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # Calculate attention weights and apply dropout
        p_attn = masked_softmax(scores, valid_lens)
        p_attn = self.dropout(p_attn)

        # Calculate the attention-weighted values
        output = torch.bmm(p_attn, values)
        # outputs: (batch_size, num_queries, value_dimension)
        return output


class MultiHeadAttentionD2L(nn.Module):
    """多头注意力.

    1. 为了避免计算代价和参数代价的大幅增长， 我们设定 𝑝𝑞=𝑝𝑘=𝑝𝑣=𝑝𝑜/ℎ 。
    2. 值得注意的是，如果我们将查询、键和值的线性变换的输出数量设置为  𝑝𝑞ℎ=𝑝𝑘ℎ=𝑝𝑣ℎ=𝑝𝑜 ， 则可以并行计算 ℎ 个头。
    3. 在下面的实现中， 𝑝𝑜 是通过参数num_hiddens指定的。
    """
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttentionD2L, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10,
                                                           4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention1 = AdditiveAttention(key_size=2,
                                   query_size=20,
                                   num_hiddens=8,
                                   dropout=0)
    res = attention1(queries, keys, values, valid_lens)
    print(res.shape)  # torch.Size([2, 1, 4])

    queries = torch.normal(0, 1, (2, 1, 2))
    attention2 = DotProductAttention(dropout=0)
    results = attention2(queries, keys, values, valid_lens)
    print(results.shape)

    # D2l.ai  MultiHeadAttentionD2L
    num_hiddens, num_heads = 100, 5
    attention3 = MultiHeadAttentionD2L(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    res = attention3(X, Y, Y, valid_lens)
    print(res.shape)
