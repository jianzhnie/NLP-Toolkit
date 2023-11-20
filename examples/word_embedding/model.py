import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../../')
from nlptoolkit.data.embeddings import PositionalEncoding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 rnn_type,
                 ntoken,
                 d_model,
                 nhid,
                 nlayers,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(d_model,
                                             nhid,
                                             nlayers,
                                             dropout=dropout)
        else:
            try:
                nonlinearity = {
                    'RNN_TANH': 'tanh',
                    'RNN_RELU': 'relu'
                }[rnn_type]
            except KeyError as e:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from e
            self.rnn = nn.RNN(d_model,
                              nhid,
                              nlayers,
                              nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != d_model:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads in the transformer.
        num_hidden_size (int): Dimensionality of the hidden layer in the feedforward network.
        num_layers (int): Number of transformer encoder layers.
        dropout (float, optional): Dropout probability. Default is 0.5.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 num_hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.5):
        super(TransformerModel, self).__init__(d_model=d_model,
                                               nhead=num_heads,
                                               dim_feedforward=num_hidden_size,
                                               num_encoder_layers=num_layers)

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.input_emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self,
                src: torch.Tensor,
                has_mask: bool = True) -> torch.Tensor:
        """Forward pass of the transformer model.

        Args:
            src (torch.Tensor): Input sequence tensor of shape (sequence_length, batch_size).
            has_mask (bool, optional): Whether to apply the source mask. Default is True.

        Returns:
            torch.Tensor: Log probabilities of the output sequence.
        """
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(
                    len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = torch.mean(output, dim=1)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
