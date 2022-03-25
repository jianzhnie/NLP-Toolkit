'''
Author: jianzhnie
Date: 2021-12-22 14:36:16
LastEditTime: 2022-03-25 17:32:20
LastEditors: jianzhnie
Description:

'''

import torch
from d2l import torch as d2l
from torch import nn

from nlptoolkit.models.lm import SkipGramModel


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs,
                                                             target,
                                                             weight=mask,
                                                             reduction='none')
        return out.mean(dim=1)


def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch
            ]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) /
                 mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                         num_noise_words)

    skip_gram = SkipGramModel(20, 4)
    res = skip_gram(torch.ones((2, 1), dtype=torch.long),
                    torch.ones((2, 4), dtype=torch.long))

    print(res)
    loss = SigmoidBCELoss()
    pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
    label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    l = loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
    print(l)

    embed_size = 100
    net = nn.Sequential(
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))

    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    lr, num_epochs = 0.002, 5
    train(net, data_iter, lr, num_epochs)
