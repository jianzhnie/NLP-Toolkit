'''
Author: jianzhnie
Date: 2021-12-30 18:21:26
LastEditTime: 2022-03-25 19:01:36
LastEditors: jianzhnie
Description:

'''

import math

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from nlptoolkit.data.embeddings.pos_embedding import PositionalEmbedding
from nlptoolkit.data.embeddings.pos_encding import PositionalEncoding
from nlptoolkit.data.embeddings.word_embedding import WordEmbedding


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransformerEncoderModel(nn.Module):
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
        self.fc = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

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
        output = self.fc(output)
        return output


class TransformerModel(nn.Module):
    """The Transformer model.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        trg_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        dim_feedforward (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing.
        bos_id (int, optional):
            The start token id and also be used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
    """
    def __init__(self,
                 src_vocab_size: int = None,
                 trg_vocab_size: int = None,
                 max_length: int = 512,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 weight_sharing: bool = False,
                 bos_id: int = 0,
                 eos_id: int = 1):
        super(TransformerModel, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.dim_feedforward = dim_feedforward
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = WordEmbedding(vocab_size=src_vocab_size,
                                                emb_dim=d_model,
                                                bos_id=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                'Vocabularies in source and target should be same for weight sharing.'
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(vocab_size=trg_vocab_size,
                                                    emb_dim=d_model,
                                                    bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                         max_length=max_length)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu')

        if weight_sharing:
            self.linear = lambda x: torch.matmul(x=x,
                                                 y=self.trg_word_embedding.
                                                 word_embedding.weight,
                                                 transpose_y=True)
        else:
            self.linear = nn.Linear(in_features=d_model,
                                    out_features=trg_vocab_size,
                                    bias_attr=False)
