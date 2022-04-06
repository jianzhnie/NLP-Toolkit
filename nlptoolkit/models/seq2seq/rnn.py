import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int,
                 dec_hid_dim: int, dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch size, emb dim]

        # outputs = [src_len, batch size,git hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        """
        decoder_hidden: batch_size * embed_dim
        encoder_outputs: seq_len * batch_size * embed_dim
        """
        src_len = encoder_outputs.shape[0]

        # repeated_decoder_hidden: batch_size * seq_len * embed_dim
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(
            1, src_len, 1)
        # encoder_outputs:  batch_size *seq_len * embed_dim
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # output: batch_size *seq_len * (embed_dim_1 +  embed_dim_2)
        output = torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        # batch_size *seq_len * attn_dim
        output = self.attn(output)
        energy = torch.tanh(output)
        # attention: batch_size *seq_len
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int,
                 dec_hid_dim: int, dropout: int, attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        # attn: batch_size * seq_len
        attn = self.attention(decoder_hidden, encoder_outputs)

        attn = attn.unsqueeze(1)

        # encoder_outputs: seq_len * batch_size * embed_dim
        #              ==> batch_size * seq_len * embed_dim
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted_encoder_rep: batch_size * seq_len * embed_dim
        #                 ===> seq_len * batch_size * embed_dim
        weighted_encoder_rep = torch.bmm(attn, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, input: Tensor, decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)
        # batch_size * seq_len * embed_dim
        embedded = self.dropout(self.embedding(input))

        # seq_len * batch_size * embed_dim
        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs)

        # seq_len * batch_size * (embed_dim1 + embed_dim2)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input,
                                          decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(
            torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        """
        src: seq_len * batch_size
        trg: seq_len * batch_size
        """
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)

        # encoder_outputs: seq_len * batch_size * embed_dim
        # hidden:  batch_size * embed_dim
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
