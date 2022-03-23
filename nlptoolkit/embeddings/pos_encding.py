'''
Author: jianzhnie
Date: 2021-12-16 15:54:45
LastEditTime: 2021-12-30 17:46:17
LastEditors: jianzhnie
Description:

'''

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError(
                'Cannot use sin/cos postional encoding with odd dim (got dim ={:d}'
                .format(emb_size))

        self.emb_size = emb_size
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2, dtype=torch.float) *
            (-math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros(max_len, emb_size)
        # 偶数位置编码
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置编码
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        # 不对位置编码求梯度
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape  [seq_len, batch_size, embedding_dim]
        """
        x = x * math.sqrt(self.emb_size)
        # 输入的词向量与位置编码相加
        x = x + self.pos_embedding[:x.size(0), :]
        return self.dropout(x)


class PositionalEncodingD2L(nn.Module):
    """位置编码."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncodingD2L, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X = position / div_term
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    vocab_size = 1000
    batch_size = 32
    seq_len = 32
    d_model = 128
    drop_out = 0
    max_len = 5000
    pe = PositionalEncoding(emb_size=d_model, dropout=drop_out)
    x = torch.from_numpy(
        np.random.randint(1, vocab_size, size=(batch_size, seq_len)))
    token = TokenEmbedding(vocab_size=vocab_size, emb_size=d_model)
    x = token(x).transpose(0, 1)
    print(x.shape)
    y = pe.forward(x)
    print(y.shape)
    plt.plot(np.arange(32), y[:, 0, 4:8].data.numpy())
    plt.legend(['dim %d' % p for p in list(range(4, 8))])
    plt.savefig('PositionalEncoding4.png')
