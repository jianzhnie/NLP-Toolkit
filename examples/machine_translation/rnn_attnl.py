import math
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nlptoolkit.datasets.nmtdataset import NMTDatasets
from nlptoolkit.models.seq2seq.rnn_attn import (Attention, Decoder, Encoder,
                                                Seq2Seq)

sys.path.append('../../')


def train(model: nn.Module, iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer, criterion: nn.Module, clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = '/home/robin/jianzh/nlp-toolkit/data'
    nmtdataset = NMTDatasets(root=root)
    src_tokens, tgt_tokens, src_vocab, tgt_vocab = nmtdataset.get_dataset_tokens(
    )

    def generate_batch(data_batch, vocab=src_vocab):
        PAD_IDX = vocab['<pad>']
        BOS_IDX = vocab['<bos>']
        EOS_IDX = vocab['<eos>']
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), de_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
            en_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), en_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch

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

    enc = Encoder(input_dim=len(src_vocab),
                  emb_dim=32,
                  enc_hid_dim=64,
                  dec_hid_dim=64,
                  dropout=0.5)

    attn = Attention(enc_hid_dim=64, dec_hid_dim=64, attn_dim=8)

    dec = Decoder(output_dim=len(tgt_vocab),
                  emb_dim=32,
                  enc_hid_dim=64,
                  dec_hid_dim=64,
                  dropout=0.5,
                  attention=attn)

    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    for epoch in range(10):

        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip=1)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}'
        )

    test_loss = evaluate(model, valid_iter, criterion)

    print(
        f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |'
    )
