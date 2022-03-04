'''
Author: jianzhnie
Date: 2021-12-16 14:18:44
LastEditTime: 2022-03-04 18:13:23
LastEditors: jianzhnie
Description:

'''

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlptoolkit.models.losses.mask_softmax import masked_softmax


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


class AdditiveAttention(nn.Module):
    """加性注意力."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # queries : torch.Size([2, 1, 20])
    # keys:     torch.Size([2, 10, 2])
    # values:  torch.Size([2, 10, 4])
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10,
                                                           4).repeat(2, 1, 1)
    valid_lens = None

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
