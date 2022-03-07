'''
Author: jianzhnie
Date: 2022-03-04 18:19:57
LastEditTime: 2022-03-07 16:16:48
LastEditors: jianzhnie
Description:

'''
# Defined in Section 5.1.3.3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from nlptoolkit.data.datasets.nlmdataset import RNNlmDataset
from nlptoolkit.data.utils.utils import (get_loader, load_reuters,
                                         save_pretrained)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 循环神经网络：这里使用LSTM
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 输出层
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        # 计算每一时刻的隐含层表示
        output, _ = self.rnn(embeds)
        output = self.linear(output)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs


if __name__ == '__main__':
    embedding_dim = 64
    context_size = 2
    hidden_dim = 128
    batch_size = 1024
    num_epoch = 10

    # 读取文本数据，构建FFNNLM训练数据集（n-grams）
    corpus, vocab = load_reuters()
    dataset = RNNlmDataset(corpus, vocab)
    data_loader = get_loader(dataset, batch_size)

    # 负对数似然损失函数，忽略pad_token处的损失
    nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
    # 构建RNNLM，并加载至device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNLM(len(vocab), embedding_dim, hidden_dim)
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]),
                            targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    save_pretrained(vocab, model.embeddings.weight.data, 'rnnlm.vec')
