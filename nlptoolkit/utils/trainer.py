import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nlptoolkit.utils.model_utils import save_model_checkpoints
from nlptoolkit.utils.train_utils import epoch_time


# Function to train the seq2seq model
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip: float = None,
    epoch: int = 100,
    device: str = 'cpu',
    log_interval: int = 10,
) -> float:
    """
    Train the seq2seq model for one epoch.

    Args:
        model (nn.Module): The seq2seq model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (optim.Optimizer): The optimizer for gradient updates.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        clip (float): Gradient clipping threshold.
        epoch (int): Current epoch number.
        log_interval (int): Log interval for printing progress.

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

        # Gradient clipping to prevent exploding gradients
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        moving_loss += loss.item()
        if idx % log_interval == 0 and idx > 0:
            cur_loss = moving_loss / log_interval
            elapsed = time.time() - start_time
            print(
                'Train: | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f}'.format(epoch, idx, len(dataloader),
                                      elapsed * 1000 / log_interval, cur_loss))
            moving_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


# Function to evaluate the seq2seq model
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.CrossEntropyLoss,
             device: str = 'cpu',
             log_interval: int = 10) -> float:
    """
    Evaluate the seq2seq model on the validation or test data.

    Args:
        model (nn.Module): The seq2seq model.
        dataloader (DataLoader): DataLoader for validation or test data.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        log_interval (int): Log interval for printing progress.

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


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    num_epochs: int = 100,
    clip: float = 0.25,
    device: str = 'cpu',
    log_interval: int = 10,
    save_model_path: str = 'rnn_nmt',
):
    """
    Train and evaluate the seq2seq model.

    Args:
        model (nn.Module): The seq2seq model.
        train_loader (DataLoader): DataLoader for the training data.
        eval_loader (DataLoader): DataLoader for the evaluation data.
        optimizer (optim.Optimizer): The optimizer for gradient updates.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        num_epochs (int): Number of training epochs.
        clip (float): Gradient clipping threshold.
        log_interval (int): Log interval for printing progress.
        save_model_path (str): Path to save the best model.

    Returns:
        None
    """
    # Initialize the best validation loss
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        # Train the model
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     clip, epoch, device, log_interval)
        # Evaluate the model on the validation set
        val_loss = evaluate(model, eval_loader, criterion, device,
                            log_interval)
        end_time = time.time()
        # Calculate elapsed time for the epoch
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # Print epoch information
        print('-' * 89)
        print(f'Epoch: {epoch:02} | Time: {epoch_mins} m {epoch_secs} s')
        print(
            f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        print(f'Val Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}')
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            save_model_checkpoints(model, epoch, save_model_path)
            best_val_loss = val_loss
    # Evaluate the model on the test set
    test_loss = evaluate(model, eval_loader, criterion, log_interval)
    print('End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
