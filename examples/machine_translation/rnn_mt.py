import math
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('../../')
from torch.utils.data.dataset import random_split

from nlptoolkit.datasets.nmtdataset import NMTDataset
from nlptoolkit.losses import MaskedSoftmaxCELoss
from nlptoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq


# Function to train the seq2seq model
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip: float = None,
    epoch: int = 100,
    log_interval: int = 10,
) -> float:
    """
    Train the seq2seq model for one epoch.

    Args:
        model (nn.Module): The seq2seq model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (optim.Optimizer): The optimizer for gradient updates.
        criterion (MaskedSoftmaxCELoss): The loss criterion.
        clip (float): Gradient clipping threshold.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    moving_loss = 0
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        src, src_len, trg, trg_len = [t.to(device) for t in batch]

        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output, trg, trg_len)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        moving_loss += loss.item()
        if idx % log_interval == 0 and idx > 0:
            cur_loss = moving_loss / log_interval
            elapsed = time.time() - start_time
            print(
                'Train: epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f}'.format(epoch, idx, len(dataloader),
                                      elapsed * 1000 / log_interval, cur_loss))
            moving_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


# Function to evaluate the seq2seq model
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.CrossEntropyLoss,
             log_interval: int = 10) -> float:
    """
    Evaluate the seq2seq model on the validation or test data.

    Args:
        model (nn.Module): The seq2seq model.
        iterator (DataLoader): DataLoader for validation or test data.
        criterion (MaskedSoftmaxCELoss): The loss criterion.

    Returns:
        float: Average evaluation loss.
    """
    model.eval()
    epoch_loss = 0
    moving_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            src, src_len, trg, trg_len = [t.to(device) for t in batch]
            output = model(src, trg)
            loss = criterion(output, trg, trg_len)
            epoch_loss += loss.item()
            moving_loss += loss.item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = moving_loss / log_interval
                print('Val: {:5d}/{:5d} batches |'
                      'loss {:5.2f} '.format(idx, len(dataloader), cur_loss))
                moving_loss = 0
    return epoch_loss / len(dataloader)


# Function to calculate elapsed time
def epoch_time(start_time: int, end_time: int) -> Tuple[int, int]:
    """
    Calculate elapsed time in minutes and seconds.

    Args:
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.

    Returns:
        Tuple[int, int]: Elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_and_evaluate(model, train_loader, eval_loader, optimizer, criterion,
                       num_epochs, clip, log_interval, save_model_path):
    # Initialize the best validation loss
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        # Train the model
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     clip, epoch, log_interval)
        # Evaluate the model on the validation set
        val_loss = evaluate(model, eval_loader, criterion, log_interval)
        end_time = time.time()
        # Calculate elapsed time for the epoch
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # Print epoch information
        print('-' * 89)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins} m {epoch_secs} s')
        print(
            f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        print(
            f'Val. Loss: {val_loss:.3f} | Val. PPL: {math.exp(val_loss):7.3f}')
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_model_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        # Evaluate the model on the test set
    test_loss = evaluate(model, eval_loader, criterion, log_interval)
    print('End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))


if __name__ == '__main__':
    # Check for available GPU, and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Define the file path for the dataset
    file_path = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    model_path = '/home/robin/work_dir/llm/nlp-toolkit/nmt/best_model.pth'
    # Create an instance of NMTDataset
    nmtdataset = NMTDataset(file_path=file_path, max_seq_len=30)
    src_vocab = nmtdataset.src_vocab
    tgt_vocab = nmtdataset.tgt_vocab

    # Split the dataset into training and validation sets
    data_train, data_val = random_split(nmtdataset, [0.7, 0.3])

    # Define batch size for DataLoader
    batch_size = 128

    # Create DataLoader for training and validation sets
    train_iter = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(data_val, batch_size=batch_size, shuffle=True)

    # Initialize the seq2seq model
    model = RNNSeq2Seq(src_vocab_size=len(src_vocab),
                       trg_vocab_size=len(tgt_vocab),
                       embed_size=32,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.5).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Function to count trainable parameters in the model
    def count_parameters(model: nn.Module) -> int:
        """
        Count the number of trainable parameters in the model.

        Args:
            model (nn.Module): The model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Initialize the loss criterion
    criterion = MaskedSoftmaxCELoss()

    train_and_evaluate(model,
                       train_iter,
                       valid_iter,
                       optimizer,
                       criterion,
                       num_epochs=10,
                       clip=0.25,
                       log_interval=10,
                       save_model_path=model_path)
