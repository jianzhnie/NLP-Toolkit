'''
Author: jianzhnie
Date: 2022-03-08 10:31:43
LastEditTime: 2022-03-08 10:47:16
LastEditors: jianzhnie
Description:

'''
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from nlptoolkit.models.transformer.gpt.config_gpt import (GPTConfig,
                                                          TrainerConfig)
from nlptoolkit.models.transformer.gpt.model_gpt import GPTModel
from nlptoolkit.models.transformer.gpt.trainer import Trainer

sys.path.append('../../../')


# run kmeans to get our codebook
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' %
              (i + 1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
    return c


class ImageDataset(Dataset):
    """wrap up the pytorch CIFAR-10 dataset into our own, which will convert
    images into sequences of integers."""
    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32 * 32) if perm is None else perm

        self.vocab_size = clusters.size(0)
        self.block_size = 32 * 32 - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3)  # flatten out all pixels
        x = x[self.perm].float(
        )  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(
            1)  # cluster assignments
        return a[:-1], a[
            1:]  # always just predict the next one in the sequence


if __name__ == '__main__':
    # pytorch helpfully makes it easy to download datasets, e.g. the common CIFAR-10 https://www.kaggle.com/c/cifar-10
    root = '/media/robin/DATA/datatsets/image_data/cifar10'
    train_data = torchvision.datasets.CIFAR10(root,
                                              train=True,
                                              transform=None,
                                              target_transform=None,
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root,
                                             train=False,
                                             transform=None,
                                             target_transform=None,
                                             download=True)
    print(len(train_data), len(test_data))

    # get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
    pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32 * 32, 3)[
        torch.randperm(32 * 32)[:5], :]
    px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()
    print(px.size())

    ncluster = 512
    with torch.no_grad():
        C = kmeans(px, ncluster, niter=8)

    print(C.size())

    # encode the training examples with our codebook to visualize how much we've lost in the discretization
    n_samples = 16
    ncol = 8
    nrow = n_samples // ncol + 1
    plt.figure(figsize=(20, 10))
    for i in range(n_samples):

        # encode and decode random data
        x, y = train_data[np.random.randint(0, len(train_data))]
        xpt = torch.from_numpy(np.array(x)).float().view(32 * 32, 3)
        ix = ((xpt[:, None, :] - C[None, :, :])**2).sum(-1).argmin(
            1)  # cluster assignments for each pixel

        # these images should look normal ideally
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(C[ix].view(32, 32, 3).numpy().astype(np.uint8))
        plt.axis('off')

    train_dataset = ImageDataset(train_data, C)
    test_dataset = ImageDataset(test_data, C)
    print(train_dataset[0][0])  # one example image flattened out into integers

    model_config = GPTConfig(train_dataset.vocab_size,
                             train_dataset.block_size,
                             n_layer=2,
                             n_head=8,
                             d_model=512)
    model = GPTModel(model_config.vocab_size, model_config.d_model,
                     model_config.n_head, model_config.n_layer,
                     model_config.block_size)
    print(model)

    tokens_per_epoch = len(train_data) * train_dataset.block_size
    train_epochs = 20  # todo run a bigger model and longer, this is tiny

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=train_epochs,
                          batch_size=8,
                          learning_rate=3e-3,
                          betas=(0.9, 0.95),
                          weight_decay=0,
                          lr_decay=True,
                          warmup_tokens=tokens_per_epoch,
                          final_tokens=train_epochs * tokens_per_epoch,
                          ckpt_path='cifar10_model.pt',
                          num_workers=4)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()
