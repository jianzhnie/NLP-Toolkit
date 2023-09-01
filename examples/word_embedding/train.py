import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from paddlenlp.datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')
from nlptoolkit.data.tokenizer import JiebaTokenizer
from nlptoolkit.data.vocab import Vocab


class BoWClassifier(nn.Module):
    """
    Bag of Words 文本分类器

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

        # 全连接层, 激活函数为tanh
        out = torch.tanh(self.fc1(embed))

        # Dropout正则化
        out = self.dropout(out)

        # 输出层
        logits = self.fc2(out)

        return logits


class TorchDataset(Dataset):
    """
    PyTorch dataset for text classification or regression tasks.

    Attributes:
        dataset: List of examples, each being a dict with keys 'text' and 'label'.
        tokenizer: Tokenizer to split text into tokens.
        vocab: Vocabulary for mapping tokens to indices.
        max_seq_len: Maximum sequence length after truncation.
        label_dtype: Data type of label tensor (int64 for classification, float32 for regression).
    """
    def __init__(self,
                 dataset: List[Dict],
                 tokenizer: JiebaTokenizer,
                 vocab: Vocab,
                 max_seq_len: int,
                 label_dtype: Optional[torch.dtype] = torch.int64):
        """
        Initialize dataset with data, tokenizer, vocab and max sequence length.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        """
        Return single example by index.

        Args:
            index: Index of example.

        Returns:
            Dictionary with 'input_ids', 'valid_length', 'label' tensors.
        """
        example = self.dataset[index]
        text = example['text']
        label = example['label']

        # Tokenize
        tokens = self.tokenizer.cut(text)

        # Numericalize
        input_ids = self.vocab.to_index(tokens)
        valid_length = len(input_ids)

        # Truncate and pad
        input_ids = self._truncate_pad(input_ids, self.max_seq_len,
                                       self.vocab.get_pad_token_id())

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        valid_length = torch.tensor(valid_length)
        label = torch.tensor(label, dtype=self.label_dtype)

        return input_ids, valid_length, label

    def _truncate_pad(self,
                      inputs: List[int],
                      max_seq_len: int,
                      padding_token_id: int = 0) -> List[int]:
        """
        Truncate and pad sequence to max sequence length.
        """
        if len(inputs) > max_seq_len:
            inputs = inputs[:max_seq_len]
        else:
            inputs = inputs + [padding_token_id] * (max_seq_len - len(inputs))
        return inputs


def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: Adam,
          loss_fn: nn.Module,
          num_epochs: int = 10,
          device: torch.device = torch.device('cpu')):

    model.to(device)
    model.train()

    for epoch in range(num_epochs):

        total_loss = 0

        for batch in train_loader:

            # Move batch tensors to device
            inputs, length, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')


def main():
    train_ds, dev_ds = load_dataset('chnsenticorp', splits=['train', 'dev'])
    num_classes = len(train_ds.label_list)
    vocab_path = '/home/robin/work_dir/llm/nlp-toolkit/data/dict.txt'
    lm_vocab = Vocab.load_vocabulary(vocab_path,
                                     unk_token='[UNK]',
                                     pad_token='[PAD]')
    lm_vocab.save_vocabulary(
        '/home/robin/work_dir/llm/nlp-toolkit/data/save_vocab.txt')

    tokenizer = JiebaTokenizer(lm_vocab)
    train_dataset = TorchDataset(train_ds, tokenizer, lm_vocab, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BoWClassifier(vocab_size=len(lm_vocab),
                          embed_dim=300,
                          num_classes=num_classes,
                          hidden_size=128,
                          dropout=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )
    model.to(device)
    train(model,
          train_dataloader,
          optimizer,
          loss_fn,
          num_epochs=20,
          device=device)


if __name__ == '__main__':
    main()
