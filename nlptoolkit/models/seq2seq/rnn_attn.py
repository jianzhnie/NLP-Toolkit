import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

sys.path.append('../../../')
from nlptoolkit.models.seq2seq.rnn_mt import RNNEncoder
from nlptoolkit.transformers.vanilla.attention import AdditiveAttention


class RNNAttentionDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super(RNNAttentionDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(input_size=embed_size + hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.)

        self.attention = AdditiveAttention(hidden_size, hidden_size,
                                           hidden_size, dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: Tensor, encoder_outputs: Tensor,
                encoder_hidden_states: Tensor,
                encoder_valid_lens) -> Tuple[Tensor]:

        # input shape: [batch_size, seq_len]
        # enc_outputs shape: [seq_len, batch_size, hidden_size]
        # enc_hidden_states shape: [num_layers, batch_size, hidden_size]
        # enc_valid_lens shape: [batch_size]

        inputs = inputs.permute(1, 0)
        embedded = self.embedding(inputs)
        # embeded shape: [seq_len, batch_size, embed_size]
        outputs = []
        seq_len = embedded.shape[0]
        for idx in range(seq_len):
            input = embedded[idx]
            # input shape: [batch_size, embed_size]
            query = torch.unsqueeze(encoder_hidden_states[-1], dim=1)
            # query shape: [batch_size, 1, hidden_size]
            context = self.attention(query, encoder_outputs, encoder_outputs,
                                     encoder_valid_lens)
            # context shape: [batch_size, 1, hidden_size]
            input = torch.unsqueeze(input, dim=1)
            # input shape: [batch_size, 1, embed_size]
            concate_features = torch.cat((context, input), dim=-1)
            concate_features = concate_features.permute(1, 0, 2)
            # concate_features shape: [batch_size, 1, embed_size + hidden_size]
            decoder_outputs, decoder_hidden_state = self.rnn(
                concate_features, encoder_hidden_states)
            # decoder_outputs shape: [1, batch_size, hidden_size]
            # decoder_hidden_state shape: [num_layers, batch_size, hidden_size]
            outputs.append(decoder_outputs)

        # attention_weights shape: [seq_len, batch_size, seq_len]
        outputs = torch.cat(outputs, dim=0)
        # outputs shape: [seq_len, batch_size, hidden_size]
        outputs = self.fc_out(outputs)
        # outputs shape: [seq_len, batch_size, vocab_size]
        outputs = outputs.permute(1, 0, 2)
        # outputs shape: [batch_size, seq_len, vocab_size]
        return outputs, decoder_hidden_state


class RNNeq2SeqModel(nn.Module):
    """
    The Seq2Seq model with attention for sequence-to-sequence learning.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
    """
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize the encoder and decoder
        self.encoder = RNNEncoder(src_vocab_size, embed_size, hidden_size,
                                  num_layers, dropout)
        self.decoder = RNNAttentionDecoder(trg_vocab_size, embed_size,
                                           hidden_size, num_layers, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).

        Returns:
            dec_outputs (torch.Tensor): Decoder outputs tensor of shape (batch_size, tgt_seq_len, trg_vocab_size).
        """
        # Encode the source sequence
        enc_outputs, enc_state = self.encoder(src)

        # Get the target sequence length and batch size
        batch_size, tgt_seq_len = tgt.shape

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs shape: [batch_size, src_seq_len, hidden_size]
        # Decode the target sequence

        dec_outputs, _ = self.decoder(tgt, enc_outputs, enc_state, None)

        return dec_outputs


if __name__ == '__main__':
    src_vocab_size = 10
    tgt_vocab_size = 20
    embed_size = 8
    hiddens_size = 16
    num_layers = 2
    batch_size = 4
    max_seq_len = 5
    encoder = RNNEncoder(src_vocab_size, embed_size, hiddens_size, num_layers)
    input = torch.LongTensor([[1, 2, 4, 5, 3], [4, 3, 2, 9, 2],
                              [1, 2, 3, 4, 4], [4, 3, 2, 1, 6]])

    target = torch.LongTensor([[1, 3, 4, 5, 3], [4, 3, 2, 9, 2],
                               [1, 2, 3, 4, 4], [4, 3, 2, 1, 6]])
    # input: [batch_size, max_seq_len]
    enc_outputs, enc_state = encoder(input)
    # enc_outputs: [max_seq_len, batch_size, hiddens_size]
    # enc_state: [num_layers, batch_size, hiddens_size]
    print(enc_outputs.shape, enc_state.shape)

    decoder = RNNAttentionDecoder(tgt_vocab_size, embed_size, hiddens_size,
                                  num_layers)
    enc_outputs = enc_outputs.permute(1, 0, 2)
    dec_outputs, decoder_hidden_state = decoder(target, enc_outputs, enc_state,
                                                None)
    print(dec_outputs.shape, decoder_hidden_state.shape)

    print('seq2seq')
    seq2seq = RNNeq2SeqModel(src_vocab_size, tgt_vocab_size, embed_size,
                             hiddens_size, num_layers)

    output = seq2seq(input, target)
    print(output.shape)
