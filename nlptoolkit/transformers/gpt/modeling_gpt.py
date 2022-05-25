import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class GPTDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation='gelu',
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True) -> None:
        super().__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiheadAttention(d_model,
                                            nhead,
                                            dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(act_dropout, inplace=True)
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None):

        if self.normalize_before:
            src2 = self.norm1(src)

        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src2 = self.dropout1(src2)
        src = src + src2

        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src = self.norm2(src)

        src = self.linear2(self.activation(self.linear1(src)))
        src2 = self.dropout2(src)
        src = src + src2

        if not self.normalize_before:
            src = self.norm1(src)

        return src
