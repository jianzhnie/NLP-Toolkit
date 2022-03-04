'''
Author: jianzhnie
Date: 2021-12-30 18:21:26
LastEditTime: 2021-12-30 18:52:08
LastEditors: jianzhnie
Description:

'''

import math

import torch
from nlptoolkit.models.layers.pos_encding import PositionalEncoding
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransformerModel(nn.Module):
    """The nn.TransformerEncoder consists of multiple layers of
    nn.TransformerEncoderLayer.

    Along with the input sequence, a square attention mask is required because the self-attention layers in nn.TransformerEncoder are only allowed to attend the
    earlier positions in the sequence. For the language  modeling task, any tokens on the future positions should be masked. To produce a probability
    distribution over output words, the output of the nn.TransformerEncoder model is passed through a linear layer followed by a log-softmax function.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 nhead: int,
                 d_ffn: int,
                 nlayers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn,
                                                 dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
