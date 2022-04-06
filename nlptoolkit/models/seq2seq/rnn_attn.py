import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, embeb_dim: int, enc_hidden_size: int,
                 num_layers: int, dec_hidden_size: int, dropout: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.embeb_dim = embeb_dim
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embeb_dim)

        self.rnn = nn.GRU(input_size=embeb_dim,
                          hidden_size=enc_hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hidden_size
        self.dec_hid_dim = dec_hidden_size

        self.attn = nn.Linear((enc_hidden_size * 2) + dec_hidden_size,
                              attn_dim)
        self.fc = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.cat((hidden, encoder_outputs), dim=2)
        energy = self.attn(energy)
        energy = torch.tanh(energy)

        # energy = [batch size, src len, dec hid dim]

        attention = self.fc(energy).squeeze(2)
        # attention = torch.sum(energy, dim=2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, enc_hidden_size: int,
                 dec_hidden_size: int, attn_dim: int, dropout: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.enc_hidden_dim = enc_hidden_size
        self.dec_hidden_dim = dec_hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.GRU((enc_hidden_size * 2) + embed_dim, dec_hidden_size)

        self.attention = Attention(enc_hidden_size=enc_hidden_size,
                                   dec_hidden_size=dec_hidden_size,
                                   attn_dim=attn_dim)
        self.fc_out = nn.Linear(
            (enc_hidden_size * 2) + dec_hidden_size + embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def _weighted_attn(self, decoder_hidden: Tensor,
                       encoder_outputs: Tensor) -> Tensor:
        # attn: batch_size * seq_len
        attn = self.attention(decoder_hidden, encoder_outputs)

        attn = attn.unsqueeze(1)

        # attn = [batch size, 1, src len]

        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = torch.bmm(attn, encoder_outputs)
        # weighted = [1, batch size, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        return weighted

    def forward(self, input: Tensor, decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        # weighted = [1, batch size, enc_hid_dim * 2]
        weighted = self._weighted_attn(decoder_hidden, encoder_outputs)

        # rnn_input = [1, batch size, (enc_hid_dim * 2) + emb dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, decoder_hidden = self.rnn(rnn_input,
                                          decoder_hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class RNNSeq2SeqAttnModel(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 enc_hidden_size,
                 dec_hidden_size,
                 attm_dim,
                 num_layers,
                 dropout=0.,
                 device='cpu'):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.device = device
        self.init_weights()

        self.encoder = RNNEncoder(src_vocab_size, embed_dim, enc_hidden_size,
                                  num_layers, enc_hidden_size, dropout)
        self.decoder = RNNDecoder(trg_vocab_size, embed_dim, enc_hidden_size,
                                  dec_hidden_size, attm_dim, dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.trg_vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token

            outputs[t] = output
            # decide if we are going to use teacher forcing or not

            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.max(1)[1]
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = (trg[t] if teacher_force else top1)

        return outputs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
