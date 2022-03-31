import math
import sys
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nlptoolkit.datasets.nmtdataset import NMTDatasets

sys.path.append('../../')


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


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
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
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


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz),
                                  device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           devic=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    losses = 0

    for src, tgt in train_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    losses = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_loader)


if __name__ == '__main__':
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = '/Users/jianzhengnie/work_dir/code_gallery/nlp-toolkit/examples/data'
    nmtdataset = NMTDatasets(root=root)
    src_tokens, tgt_tokens, src_vocab, tgt_vocab = nmtdataset._build_tokens()

    def generate_batch(data_batch, vocab=src_vocab):
        PAD_IDX = vocab['<pad>']
        BOS_IDX = vocab['<bos>']
        EOS_IDX = vocab['<eos>']
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), src_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
            tgt_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), tgt_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def get_dataloader(train_data, val_data, batch_size=128):

        train_iter = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=generate_batch)
        valid_iter = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=generate_batch)
        return train_iter, valid_iter

    data_train = nmtdataset.get_tensor_dataset(src_tokens,
                                               tgt_tokens,
                                               train=True)
    data_val = nmtdataset.get_tensor_dataset(src_tokens,
                                             tgt_tokens,
                                             train=False)

    train_iter, valid_iter = get_dataloader(data_train,
                                            data_val,
                                            batch_size=128)

    transformer = Seq2SeqTransformer(src_vocab_size=len(src_vocab),
                                     tgt_vocab_size=len(tgt_vocab),
                                     num_encoder_layers=1,
                                     num_decoder_layers=1,
                                     emb_size=32,
                                     nhead=1,
                                     dim_feedforward=32)

    transformer = transformer.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=0.0001,
                                 betas=(0.9, 0.98),
                                 eps=1e-9)

    for epoch in range(1, 10 + 1):
        start_time = timer()
        train_loss = train_epoch(transformer,
                                 train_iter,
                                 optimizer,
                                 device=device)
        end_time = timer()
        val_loss = evaluate(transformer, valid_iter, device=device)
        print((
            f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, '
            f'Epoch time = {(end_time - start_time):.3f}s'))
