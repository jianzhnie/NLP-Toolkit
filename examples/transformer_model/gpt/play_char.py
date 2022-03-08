'''
Author: jianzhnie
Date: 2022-03-07 17:23:31
LastEditTime: 2022-03-08 10:15:37
LastEditors: jianzhnie
Description:

'''
import sys

import torch
from torch.utils.data import Dataset

from nlptoolkit.models.transformer.gpt.config_gpt import (GPTConfig,
                                                          TrainerConfig)
from nlptoolkit.models.transformer.gpt.model_gpt import GPTModel
from nlptoolkit.models.transformer.gpt.trainer import Trainer
from nlptoolkit.models.transformer.gpt.utils import sample

sys.path.append('../../../')


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == '__main__':

    block_size = 128  # spatial extent of the model for its context

    text = open('/home/robin/jianzh/nlp-toolkit/data/timemachine.txt',
                'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(
        text, block_size)  # one line of poem is roughly 50 characters

    model_config = GPTConfig(train_dataset.vocab_size,
                             train_dataset.block_size,
                             n_layer=2,
                             n_head=8,
                             d_model=512)
    model = GPTModel(model_config.vocab_size, model_config.d_model,
                     model_config.n_head, model_config.n_layer,
                     model_config.block_size)
    print(model)
    train_config = TrainerConfig(max_epochs=2,
                                 batch_size=32,
                                 learning_rate=6e-4,
                                 lr_decay=True,
                                 warmup_tokens=512 * 20,
                                 final_tokens=2 * len(train_dataset) *
                                 block_size,
                                 num_workers=4)
    trainer = Trainer(model, train_dataset, None, train_config)
    trainer.train()

    context = 'the time machine'
    x = torch.tensor([train_dataset.stoi[s] for s in context],
                     dtype=torch.long)[None, ...].to(trainer.device)
    y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)
