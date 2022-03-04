'''
Author: jianzhnie
Date: 2021-12-23 17:02:12
LastEditTime: 2021-12-23 17:05:53
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter,
                 num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim,
                                num_filter,
                                filter_size,
                                padding=1)
        self.activate = nn.ReLU()
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        """
        inputs: batch * seq_length
        step1 : embedding, batch * seq_length ==>  batch * seq_length * embedding_dim
                    ==> batch *  embedding_dim * seq_length
        step2 : conv1d   batch *  embedding_dim * seq_length ==>  batch *  embedding_dim * seq_length
        step3 : relu()
        step4 : Linear()
        """
        embedding = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat(
            (self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs
        ],
                             dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
