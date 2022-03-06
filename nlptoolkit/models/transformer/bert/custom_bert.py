'''
Author: jianzhnie
Date: 2021-12-28 11:57:58
LastEditTime: 2022-01-05 10:19:52
LastEditors: jianzhnie
Description:

'''

import copy
import sys

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_

from nlptoolkit.models.attention.attention import MultiHeadAttention
from nlptoolkit.models.layers.layers import FeedForwardNetwork

sys.path.append('../../../../')


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class AddNorm(nn.Module):
    """残差连接后进行层规范化."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class BertEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        max_len=1000,
        type_vocab_size=2,
        dropout_prob=0.1,
        pad_token_id=0,
    ):
        super().__init__()

        # padding_idx (int, optional): If given, pads the output with the embedding vector at
        # `padding_idx` (initialized to zero) whenever it encounter the index
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, tokens=None, segments=None):
        seq_len = tokens.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        # (seq_len,) -> (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(tokens)
        inputs_embeds = self.word_embeddings(tokens)
        token_type_embeddings = self.token_type_embeddings(segments)
        position_embeddings = self.position_embeddings(pos)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, num_hiddens=768, num_heads=12, d_ffn=2048, dropout=0.1):
        super(BertEncoder, self).__init__()
        self.self_attn = MultiHeadAttention(num_hiddens,
                                            num_heads,
                                            dropout=dropout)
        # Implementation of Feedforward model
        self.ffn = FeedForwardNetwork(num_hiddens, d_ffn)
        self.addnorm1 = AddNorm(normalized_shape=num_hiddens, dropout=dropout)
        self.addnorm2 = AddNorm(normalized_shape=num_hiddens, dropout=dropout)

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, enc_self_attn = self.self_attn(src, src, src, mask)[0]
        src = self.addnorm1(src, src2)
        src2 = self.ffn(src)
        src = self.addnorm2(src, src2)
        return src, enc_self_attn


class BertModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_hiddens: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 ffn_num_hiddens: int = 2048,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(BertModel, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embeddings = BertEmbedding(vocab_size=vocab_size,
                                        hidden_size=num_hiddens)
        self.encoder_layer = BertEncoder(num_hiddens=num_hiddens,
                                         num_heads=num_heads,
                                         ffn_num_hiddens=ffn_num_hiddens,
                                         dropout=dropout)
        self.transformer_blocks = _get_clones(self.encoder_layer, num_layers)
        self._reset_parameters()

    def forward(self, x: Tensor, segments: Tensor) -> Tensor:
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        x = self.embeddings(x, segments)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x, enc_self_attn = transformer(x, mask)

        return x, enc_self_attn

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class BERTLM(nn.Module):
    """BERT Language Model Next Sentence Prediction Model + Masked Language
    Model."""
    def __init__(self, bert: BertModel, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.num_hiddens)
        self.mask_lm = MaskedLanguageModel(vocab_size, self.bert.hidden,
                                           self.bert.hidden)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """
    def __init__(self, num_inputs):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(num_inputs, 2)

    def forward(self, x):

        return self.linear(x[:, 0])


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        """
        param:
        - hidden: output size of BERT model
        - vocab_size: total vocab size
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # batch_size x len_q x len_k
    return pad_attn_mask.expand(batch_size, len_q, len_k)


if __name__ == '__main__':
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 12
    bert = BertModel(vocab_size, num_hiddens)
    encoder = BertEncoder(num_hiddens, num_heads)
    print(bert)
    attn = MultiHeadAttention(768, 12)
