import argparse
import math
import time

import torch
import torch.nn as nn
from model import RNNModel, TransformerModel
from torch.utils.data import DataLoader

from data import WikiTextDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model'
    )
    parser.add_argument('--data',
                        type=str,
                        default='../../data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument(
        '--model',
        type=str,
        default='LSTM',
        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--d_model',
                        type=int,
                        default=200,
                        help='size of word embeddings')
    parser.add_argument('--num_hidden_size',
                        type=int,
                        default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='number of layers')
    parser.add_argument('--lr',
                        type=float,
                        default=20,
                        help='initial learning rate')
    parser.add_argument('--clip',
                        type=float,
                        default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        metavar='N',
                        help='batch size')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied',
                        action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda',
                        action='store_true',
                        default=False,
                        help='use CUDA')
    parser.add_argument('--log-interval',
                        type=int,
                        default=200,
                        metavar='N',
                        help='report interval')
    parser.add_argument('--save',
                        type=str,
                        default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export',
                        type=str,
                        default='',
                        help='path to export the final model in onnx format')
    parser.add_argument(
        '--num_heads',
        type=int,
        default=2,
        help='the number of heads in the encoder of the transformer model')
    args = parser.parse_args()
    return args


def train_one_epoch(
    model,
    train_data,
    criterion,
    lr,
    epoch,
    vocab_size,
    args,
):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for idx, batch in enumerate(train_data):
        inputs, targets = batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(inputs)
            output = output.view(-1, vocab_size)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, idx, len(train_data), lr,
                    elapsed * 1000 / args.log_interval, cur_loss,
                    math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return total_loss


def evaluate(model, eval_data, criterion, vocab_size, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for idx, batch in enumerate(eval_data):
            data, targets = batch
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, vocab_size)
                cur_loss = criterion(output, targets)

            total_loss += cur_loss.item()
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = total_loss / args.log_interval
                print('Val | {:5d}/{:5d} batches |'
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          idx, len(eval_data), cur_loss, math.exp(cur_loss)))
                total_loss = 0
    return total_loss / len(eval_data)


def train_and_evaluate(model, train_data, eval_data, criterion, lr, epoch,
                       vocab_size, args):
    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_one_epoch(model, train_data, criterion, lr, epoch, vocab_size,
                        args)
        val_loss = evaluate(model, eval_data, criterion, vocab_size, args)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch,
                                         (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    print('=' * 89)


def main():
    args = parse_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                'WARNING: You have a CUDA device, so you should probably run with --cuda.'
            )

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the dataset
    train_dataset = WikiTextDataset(args.data, 'train')
    valid_dataset = WikiTextDataset(args.data, 'valid')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True)
    eval_loader = DataLoader(valid_dataset,
                             batch_size=args.batch_size,
                             collate_fn=train_dataset.collate_fn,
                             shuffle=True)

    vocab_size = len(train_dataset.vocab)
    if args.model == 'Transformer':
        model = TransformerModel(vocab_size,
                                 d_model=args.d_model,
                                 num_heads=args.num_heads,
                                 num_hidden_size=args.num_hidden_size,
                                 num_layers=args.num_layers)
    else:
        model = RNNModel(args.model, vocab_size, args.d_model,
                         args.num_hidden_size, args.num_layers, args.dropout,
                         args.tied).to(device)
    criterion = nn.NLLLoss()

    train_and_evaluate(model,
                       train_loader,
                       eval_loader,
                       criterion=criterion,
                       lr=args.lr,
                       epoch=args.epochs,
                       vocab_size=vocab_size,
                       args=args)


if __name__ == '__main__':

    main()
