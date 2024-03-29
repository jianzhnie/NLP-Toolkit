import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 kernel_size,
                 dropout,
                 max_length=100,
                 device='cpu'):
        super().__init__()

        assert kernel_size % 2 == 1, 'Kernel size must be odd!'

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        self.emb2hid = nn.Linear(embed_dim, hidden_size)
        self.hid2emb = nn.Linear(hidden_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=2 * hidden_size,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size,
                                                           1).to(self.device)

        # pos = [0, 1, 2, 3, ..., src len - 1]

        # pos = [batch size, src len]

        # embed tokens and positions
        tok_embedded = self.token_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = pos_embedded = [batch size, src len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, src len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src len]

        # begin convolutional blocks...

        for i, conv in enumerate(self.convs):

            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # ...end convolutional blocks

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        # combined = [batch size, src len, emb dim]

        return conved, combined


class CNNDecoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 max_length=100,
                 device=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        self.emb2hid = nn.Linear(embed_dim, hidden_size)
        self.hid2emb = nn.Linear(hidden_size, embed_dim)

        self.attn_hid2emb = nn.Linear(hidden_size, embed_dim)
        self.attn_emb2hid = nn.Linear(embed_dim, hidden_size)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=2 * hidden_size,
                      kernel_size=kernel_size) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved,
                            encoder_combined):

        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved +
                             attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):

        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size,
                                                           1).to(self.device)

        # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.token_embedding(trg)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):

            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size -
                                  1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)

            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, trg len]

            # calculate attention
            attention, conved = self.calculate_attention(
                embedded, conved, encoder_conved, encoder_combined)

            # attention = [batch size, trg len, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        # output = [batch size, trg len, output dim]

        return output, attention


class CNNSeq2Seq(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 device='cpu'):
        super().__init__()

        self.encoder = CNNEncoder(vocab_size=src_vocab_size,
                                  embed_dim=embed_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  kernel_size=kernel_size,
                                  dropout=dropout,
                                  device=device)

        self.decoder = CNNDecoder(vocab_size=trg_vocab_size,
                                  embed_dim=embed_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  kernel_size=kernel_size,
                                  dropout=dropout,
                                  trg_pad_idx=trg_pad_idx,
                                  device=device)

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        # positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        # each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention
