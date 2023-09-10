import random

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
    RNN Encoder module for sequence-to-sequence learning.

    Args:
        vocab_size (int): The size of the vocabulary.
        embed_size (int): The size of word embeddings.
        hidden_size (int): The size of the GRU's hidden state.
        num_layers (int): The number of GRU layers.
        dropout (float): Dropout probability (default: 0.5).

    """
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer to convert input tokens to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU layer with optional dropout
        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_param)

    def init_param(self, module: nn.Module):
        """
        Initialize weights for sequence-to-sequence learning.

        Args:
            module (nn.Module): The module for weight initialization.

        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.GRU):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    nn.init.xavier_uniform_(param)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            src (torch.Tensor): Input sequences with shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Output sequences with shape [seq_len, batch_size, hidden_size].
            torch.Tensor: Final hidden state with shape [num_layers, batch_size, hidden_size].

        """
        # Permute input to [seq_len, batch_size]
        src = src.permute(1, 0)

        # Pass input through embedding layer
        embedded = self.embedding(src)  # [seq_len, batch_size, embed_size]

        # Pass through GRU layers
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden


class RNNDecoder(nn.Module):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 num_layers,
                 dropout=0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(embed_size + hidden_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers > 1 else 0.)

        self.fc_out = nn.LazyLinear(vocab_size)

        self.dropout = nn.Dropout(dropout)

        self.apply(self.init_param)

    def init_param(self, module: nn.Module):
        """
        Initialize weights for sequence-to-sequence learning.

        Args:
            module (nn.Module): The module for weight initialization.

        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.GRU):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    nn.init.xavier_uniform_(param)

    def forward(self, input, hidden, context):
        input = input.permute(1, 0)
        embed = self.embedding(input)
        embed_context = torch.cat((embed, context), dim=2)
        outputs, hidden = self.rnn(embed_context, hidden)
        outputs = self.fc_out(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden shape: (num_layers, batch_size, num_hiddens)
        return outputs, hidden


class RNNSeq2Seq(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_size,
                 hidden_size,
                 num_layers,
                 dropout=0.,
                 device='cpu'):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.encoder = RNNEncoder(src_vocab_size, embed_size, hidden_size,
                                  num_layers, dropout)
        self.decoder = RNNDecoder(trg_vocab_size, embed_size, hidden_size,
                                  num_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.trg_vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        _, context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


if __name__ == '__main__':
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 5
    encoder = RNNEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    input = torch.LongTensor([[1, 2, 4, 5, 3], [4, 3, 2, 9, 2],
                              [1, 2, 3, 4, 4], [4, 3, 2, 1, 6]])
    enc_outputs, enc_state = encoder(input)
    print(enc_outputs.shape, enc_state.shape)
