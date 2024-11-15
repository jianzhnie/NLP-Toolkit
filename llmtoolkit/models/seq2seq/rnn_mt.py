from typing import Tuple

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """RNN Encoder module for sequence-to-sequence learning.

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
        """Initialize weights for sequence-to-sequence learning.

        Args:
            module (nn.Module): The module for weight initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.GRU):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    nn.init.xavier_uniform_(param)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

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
        # outputs: [seq_len, batch_size, hidden_size]
        # hidden:  [num_layers, batch_size, hidden_size]
        return outputs, hidden


class RNNDecoder(nn.Module):
    """The RNN decoder for sequence-to-sequence learning.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Size of word embeddings.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout probability (default: 0.5).
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU RNN layer
        self.rnn = nn.GRU(embed_size + hidden_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers > 1 else 0.)

        # Linear layer for output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize model parameters
        self.apply(self.init_param)

    def init_param(self, module: nn.Module):
        """Initialize weights for sequence-to-sequence learning.

        Args:
            module (nn.Module): The module for weight initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.GRU):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    nn.init.xavier_uniform_(param)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor,
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RNN Decoder.

        Args:
            input (torch.Tensor): Input tokens (batch_size, seq_len).
            hidden (torch.Tensor): Initial hidden state (num_layers, batch_size, hidden_size).
            context (torch.Tensor): Context information (batch_size, seq_len, context_size).

        Returns:
            torch.Tensor: Output logits (batch_size, seq_len, vocab_size).
            torch.Tensor: Updated hidden state (num_layers, batch_size, hidden_size).
        """
        input = input.permute(1, 0)  # Transpose for RNN input
        embed = self.embedding(input)
        # embed: [seq_len, batch_size, embed_size]
        # context: [seq_len, batch_size, context_size]
        embed_context = torch.cat((embed, context), dim=2)
        # embed_context: [seq_len, batch_size, embed_size + context_size]
        outputs, hidden = self.rnn(embed_context, hidden)
        # outputs: [seq_len, batch_size, hidden_size]
        outputs = self.fc_out(outputs)
        # outputs: [seq_len, batch_size, vocab_size]
        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, vocab_size]
        return outputs, hidden


class RNNSeq2Seq(nn.Module):
    """Sequence-to-Sequence (Seq2Seq) model using an encoder-decoder
    architecture.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        trg_vocab_size (int): Size of the target vocabulary.
        embed_size (int): Size of word embeddings.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of RNN layers.
        dropout (float, optional): Dropout probability (default: 0.0).
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
        self.decoder = RNNDecoder(trg_vocab_size, embed_size, hidden_size,
                                  num_layers, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Seq2Seq model.

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

        # Use the last encoder output as context
        context = enc_outputs[-1]

        # Broadcast context to (tgt_seq_len, batch_size, hidden_size)
        context = context.repeat(tgt_seq_len, 1, 1)

        # Decode the target sequence
        dec_outputs, _ = self.decoder(tgt, enc_state, context)

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

    decoder = RNNDecoder(tgt_vocab_size, embed_size, hiddens_size, num_layers)
    # context: [batch_size, hiddens_size]
    context = enc_outputs[-1]
    # Broadcast context to (max_seq_len, batch_size, hiddens_size)
    context = context.repeat(max_seq_len, 1, 1)
    dec_outputs, state = decoder(target, enc_state, context)
    print(dec_outputs.shape, state.shape)

    print('seq2seq')
    seq2seq = RNNSeq2Seq(src_vocab_size, tgt_vocab_size, embed_size,
                         hiddens_size, num_layers)

    output = seq2seq(input, target)
    print(output.shape)
