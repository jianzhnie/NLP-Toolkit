'''
Author: jianzhnie
Date: 2021-12-21 18:30:42
LastEditTime: 2022-01-04 17:44:04
LastEditors: jianzhnie
Description:

'''
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 线性变换：输入层->隐含层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # 使用ReLU激活函数
        self.activate = F.relu
        # 线性变换：隐含层->输出层
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs, dim=1)  # 获得每个输入属于某一类别的概率
        return probs


class MLPEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        # 使用ReLU激活函数
        self.activate = F.relu
        # 线性变换：激活层->输出层
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # 将序列中多个embedding进行聚合（此处是求平均值）
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        # 获得每个序列属于某一类别概率的对数值
        probs = F.log_softmax(outputs, dim=1)
        return probs


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络."""
    def __init__(self,
                 ffn_num_input,
                 ffn_num_hiddens,
                 ffn_num_outputs,
                 dropout=0.1,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # x = [batch size, seq len, ffn_num_input]
        X = self.relu(self.dense1(X))
        # x = [batch size, seq len, ffn_num_hiddens]
        X = self.dropout(X)
        X = self.dense2(X)
        # x = [batch size, seq len, hid dim]

        return X


class FeedForwardNetwork(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()

        self.dense1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


if __name__ == '__main__':
    mlp = MLP(input_dim=4, hidden_dim=5, num_class=2)
    inputs = torch.rand(3, 4)  # 输入形状为(3, 4)的张量，其中3表示有3个输入，4表示每个输入的维度
    probs = mlp(inputs)  # 自动调用forward函数
    print(probs)  # 输出3个输入对应输出的概率

    ffn = PositionWiseFFN(4, 4, 8)
    print(ffn(torch.ones((2, 3, 4))).shape)
