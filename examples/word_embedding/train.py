import sys

import torch
import torch.nn as nn
from paddlenlp.data.vocab import Vocab
from paddlenlp.datasets import load_dataset

from nlptoolkit.data.vocab import Vocab


class BoWClassifier(nn.Module):
    """
    Bag of Words文本分类器

    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these representations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size (obj:`int`): The vocabulary size.
        emb_dim (obj:`int`, optional, defaults to 300):  The embedding dimension.
        hidden_size (obj:`int`, optional, defaults to 128): The first full-connected layer hidden size.
        fc_hidden_size (obj:`int`, optional, defaults to 96): The second full-connected layer hidden size.
        num_classes (obj:`int`): All the labels that the data has.

    参数:
    - vocab_size: 词表大小
    - embed_dim: 嵌入维度
    - num_classes: 分类数
    - hidden_size: 第一个全连接层隐藏层大小
    - dropout: dropout概率

    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_classes: int,
                 hidden_size: int = 128,
                 dropout: float = 0.5):
        super().__init__()

        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # 全连接层
        self.fc = nn.Linear(embed_dim, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, text: torch.Tensor):
        """
        前向传播过程

        参数:
        - text: 输入文本,形状为 [batch_size, seq_len]

        返回:
        - output: 分类logits,形状为 [batch_size, num_classes]

        """

        # 获取词嵌入表示
        embed = self.embed(text)

        # 对序列dimension求平均,得到文本表示
        embed = torch.mean(embed, dim=1)

        # 全连接层 extracted_embed
        extracted_embed = torch.tanh(self.fc(embed))

        # Dropout正则化
        extracted_embed = self.dropout(extracted_embed)

        # 输出层
        output = self.out(extracted_embed)

        return output


def main():
    train_ds, dev_ds = load_dataset('chnsenticorp', splits=['train', 'dev'])
    print(train_ds)

    vocab_path = '/home/robin/work_dir/llm/nlp-toolkit/data/dict.txt'
    lm_vocab = Vocab.load_vocabulary(vocab_path)
    print(lm_vocab)
    print(len(lm_vocab))

    lm_vocab.save_vocabulary(
        '/home/robin/work_dir/llm/nlp-toolkit/data/vocab.txt')
    model = BoWClassifier(vocab_size=len(lm_vocab),
                          embed_dim=128,
                          num_classes=len(train_ds.label_list),
                          hidden_size=128,
                          dropout=0.5)
    print(model)
    return


if __name__ == '__main__':
    main()
