import sys

import torch
import torch.nn as nn
from paddlenlp.datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')
from nlptoolkit.data.tokenizer import JiebaTokenizer
from nlptoolkit.data.vocab import Vocab, truncate_pad


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
        self.fc1 = nn.Linear(embed_dim, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)

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
        extracted_embed = torch.tanh(self.fc1(embed))

        # Dropout正则化
        extracted_embed = self.dropout(extracted_embed)

        # 输出层
        output = self.fc2(extracted_embed)

        return output


class TorchDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer: JiebaTokenizer,
                 vocab: Vocab,
                 max_seq_len: int,
                 pad_token_id: int = 0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __getitem__(self, index):
        example = self.dataset[index]
        text = example['text']
        label = int(example['label'])
        tokens = self.tokenizer.cut(text)

        input_ids = self.vocab.to_index(tokens)
        valid_length = torch.tensor(len(input_ids))

        input_ids = truncate_pad(inputs=input_ids,
                                 max_seq_len=self.max_seq_len,
                                 padding_token=self.pad_token_id)

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        label = torch.tensor(label, dtype=torch.int64)

        return input_ids, valid_length, label

    def __len__(self):
        return len(self.dataset)


def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: Adam,
          loss_fn: None,
          num_epoch: int = 10,
          device: str = 'cpu'):

    for epoch in range(num_epoch):
        total_loss = 0
        for batch in train_loader:
            input_ids, valid_length, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')


def main():
    train_ds, dev_ds = load_dataset('chnsenticorp', splits=['train', 'dev'])
    num_classes = len(train_ds.label_list)
    vocab_path = '/home/robin/work_dir/llm/nlp-toolkit/data/dict.txt'
    lm_vocab = Vocab.load_vocabulary(vocab_path, unk_token='[UNK]')

    tokenizer = JiebaTokenizer(lm_vocab)
    train_dataset = TorchDataset(train_ds, tokenizer, lm_vocab, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BoWClassifier(vocab_size=len(lm_vocab),
                          embed_dim=128,
                          num_classes=num_classes,
                          hidden_size=128,
                          dropout=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.to(device)
    train(model,
          train_dataloader,
          optimizer,
          loss_fn,
          num_epoch=10,
          device=device)


if __name__ == '__main__':
    main()
