import math
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('../../')
from torch.utils.data.dataset import random_split

from nlptoolkit.datasets.nmtdataset import NMTDataset
from nlptoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq


def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
          criterion: nn.Module, clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)
        print(output.shape)
        print(trg.shape)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module):

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
    device = 'cpu'
    file_path = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    nmtdataset = NMTDataset(file_path=file_path, max_seq_len=30)
    src_vocab = nmtdataset.src_vocab
    tgt_vocab = nmtdataset.tgt_vocab
    data_train, data_val = random_split(nmtdataset, [0.7, 0.3])
    batch_size = 128
    train_iter = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_iter = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=True,
    )
    model = RNNSeq2Seq(src_vocab_size=len(src_vocab),
                       trg_vocab_size=len(tgt_vocab),
                       embed_size=32,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.5,
                       device=device).to(device)
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
