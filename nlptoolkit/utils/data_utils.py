'''
Author: jianzhnie
Date: 2021-12-17 12:13:36
LastEditTime: 2022-01-05 15:25:13
LastEditors: jianzhnie
Description:

'''
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader

from nlptoolkit.data.vocab import Vocab

sys.path.append('../../../')

# Constants
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def truncate_pad(inputs: List[int],
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


def load_ptb_data(data_dir, split='train'):
    """Penn Tree Bank（PTB）。该语料库取自“华尔街日报”的文章，分为训练集、验证集和测试集。
    在原始格式中，文本文件的每一行表示由空格分隔的一句话, 函数 `read_ptb_data` 将PTB数据集加载到文本行的列表中
    data_url: http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip
    """
    # Readthetrainingset.
    file_path = os.path.join(data_dir, 'ptb.{}.txt'.format(split))

    with open(file_path) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def load_sentence_polarity():
    from nltk.corpus import sentence_polarity

    vocab = Vocab(sentence_polarity.sents())
    train_data = [
        (vocab.to_ids(sentence), 0)
        for sentence in sentence_polarity.sents(categories='pos')[:4000]
    ] + [(vocab.to_ids(sentence), 1)
         for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    test_data = [
        (vocab.to_ids(sentence), 0)
        for sentence in sentence_polarity.sents(categories='pos')[4000:]
    ] + [(vocab.to_ids(sentence), 1)
         for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    # padding: True
    # seqs: False
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.shape[0], max_len) > lengths.unsqueeze(1)
    mask = mask.type(torch.bool)
    return mask


def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))
    vocab = Vocab(sents, reserved_tokens=['<pad>'])
    tag_vocab = Vocab(postags)
    train_data = [(vocab.to_ids(sentence), tag_vocab.to_ids(tags))
                  for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.to_ids(sentence), tag_vocab.to_ids(tags))
                 for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab


def load_reuters():
    from nltk.corpus import reuters
    text = reuters.sents()
    # lowercase (optional)
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab(tokens=text,
                  reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    corpus = [vocab.to_ids(sentence) for sentence in text]

    return corpus, vocab


def save_pretrained(vocab, embeds, save_path):
    """Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line."""
    with open(save_path, 'w') as writer:
        writer.write(f'{embeds.shape[0]} {embeds.shape[1]}\n')
        for idx, token in enumerate(vocab.idx_to_token):
            vec = ' '.join(['{:.4f}'.format(x) for x in embeds[idx]])
            writer.write(f'{token} {vec}\n')
    print(f'Pretrained embeddings saved to: {save_path}')


def load_pretrained(load_path):
    with open(load_path, 'r') as fin:
        # Optional: depending on the specific format of pretrained vector file
        n, d = map(int, fin.readline().split())
        tokens = []
        embeds = []
        for line in fin:
            line = line.rstrip().split(' ')
            token, embed = line[0], list(map(float, line[1:]))
            tokens.append(token)
            embeds.append(embed)
        vocab = Vocab(tokens)
        embeds = torch.tensor(embeds, dtype=torch.float)
    return vocab, embeds


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             collate_fn=dataset.collate_fn,
                             shuffle=shuffle)
    return data_loader


if __name__ == '__main__':
    train_data, test_data, vocab = load_sentence_polarity()
    print(train_data[:10])

    train_data, test_data, vocab, tag_vocab = load_treebank()
    print(train_data[:10])
    print(test_data[:10])
