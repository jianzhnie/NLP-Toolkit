'''
Author: jianzhnie
Date: 2021-12-30 18:21:26
LastEditTime: 2022-03-25 19:01:36
LastEditors: jianzhnie
Description:

'''

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

from nlptoolkit.data.embeddings.pos_embedding import PositionalEmbedding
from nlptoolkit.data.embeddings.pos_encding import PositionalEncoding
from nlptoolkit.data.embeddings.token_embedding import TokenEmbedding


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
        tgt_vocab_size (int):
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
                 tgt_vocab_size: int = None,
                 max_length: int = 512,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 bos_id: int = 0,
                 eos_id: int = 1):
        super(TransformerModel, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.dim_feedforward = dim_feedforward
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = TokenEmbedding(vocab_size=src_vocab_size,
                                                 emb_dim=d_model)
        self.src_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)

        self.tgt_word_embedding = TokenEmbedding(vocab_size=tgt_vocab_size,
                                                 emb_dim=d_model)
        self.tgt_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu')

        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src_word, tgt_word):
        r"""
        The Transformer forward methods. The input are source/target sequences, and
        returns logits.

        Args:
            src_word (Tensor):
                The ids of source sequences words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
            tgt_word (Tensor):
                The ids of target sequences words. It is a tensor with shape
                `[batch_size, target_sequence_length]` and its data type can be
                int or int64.

        Returns:
            Tensor:
                Output tensor of the final layer of the model whose data
                type can be float32 or float64 with shape
                `[batch_size, sequence_length, vocab_size]`.
        """
        src_max_len = src_word.shape[-1]
        tgt_max_len = tgt_word.shape[-1]
        src_slf_attn_mask = torch.cast(
            src_word == self.bos_id,
            dtype=torch.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        src_slf_attn_mask.stop_gradient = True
        tgt_slf_attn_mask = self.transformer.generate_square_subsequent_mask(
            tgt_max_len)
        tgt_slf_attn_mask.stop_gradient = True
        tgt_src_attn_bias = src_slf_attn_mask

        src_pos = torch.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * torch.arange(
                start=0, end=src_max_len, dtype=src_word.dtype)
        tgt_pos = torch.cast(
            tgt_word != self.bos_id, dtype=src_word.dtype) * torch.arange(
                start=0, end=tgt_max_len, dtype=tgt_word.dtype)

        src_word_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_word_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb

        tgt_word_emb = self.tgt_word_embedding(tgt_word)
        tgt_pos_emb = self.tgt_pos_embedding(tgt_pos)
        tgt_emb = tgt_word_emb + tgt_pos_emb
        dec_input = F.dropout(
            tgt_emb, p=self.dropout,
            training=self.training) if self.dropout else tgt_emb

        dec_output = self.transformer(enc_input,
                                      dec_input,
                                      src_mask=src_slf_attn_mask,
                                      tgt_mask=tgt_slf_attn_mask,
                                      memory_mask=tgt_src_attn_bias)

        predict = self.linear(dec_output)

        return predict


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size,
                                                      dropout=dropout)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.src_tok_emb(src)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.tgt_tok_emb(trg)
        tgt_emb = self.positional_encoding(tgt_emb)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
