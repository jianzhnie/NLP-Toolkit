'''
Author: jianzhnie
Date: 2021-12-20 09:39:27
LastEditTime: 2021-12-20 09:59:18
LastEditors: jianzhnie
Description:

'''
import re
import sys

from d2l import torch as d2l

from nlptoolkit.data.vocab import Vocab, tokenize

sys.path.append('../../')


def read_time_machine():
    with open(d2l.download('time_machine', cache_dir='./data'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus


if __name__ == '__main__':
    lines = read_time_machine()
    print(lines[0])
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])
    corpus = load_corpus_time_machine()
    print(corpus)
