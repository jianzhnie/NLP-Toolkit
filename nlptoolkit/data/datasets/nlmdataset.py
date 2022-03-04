'''
Author: jianzhnie
Date: 2022-01-05 09:40:22
LastEditTime: 2022-03-04 17:17:59
LastEditors: jianzhnie
Description:

'''
from collections import defaultdict

from tqdm.auto import tqdm

import torch
from nlptoolkit.data.utils.utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RNNlmDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            # 模型输入：BOS_TOKEN, w_1, w_2, ..., w_n
            input = [self.bos] + sentence
            # 模型输出：w_1, w_2, ..., w_n, EOS_TOKEN
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets,
                               batch_first=True,
                               padding_value=self.pad)
        return (inputs, targets)


class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            # 插入句首句尾符号
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i - context_size:i]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)


class CbowDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 模型输入：左右分别取context_size长度的上下文
                context = sentence[(i - context_size):i] + sentence[
                    (i + 1):(i + context_size + 1)]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出：一定窗口大小内的上下文
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[
                    (i + 1):(right_context_index + 1)]
                self.data.extend([(w, c) for c in context])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)


class NegativeSampleingSkipGramDataset(Dataset):
    """Negative Sampleing for Skip-Gram Dataset."""
    def __init__(self,
                 corpus,
                 vocab,
                 context_size=2,
                 n_negatives=5,
                 ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：(w, context) ；输出为0/1，表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[
                    (i + 1):(right_context_index + 1)]
                context += [self.pad] * (2 * context_size - len(context))
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用uniform分布
        self.ns_dist = ns_dist if ns_dist is not None else torch.ones(
            len(vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []
        # 对batch内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context
            ns_dist = self.ns_dist.index_fill(0, contexts[i], .0)
            neg_contexts.append(
                torch.multinomial(ns_dist,
                                  self.n_negatives * context_size,
                                  replacement=True))
        neg_contexts = torch.stack(neg_contexts, dim=0)
        return words, contexts, neg_contexts


class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        # 记录词与上下文在给定语料中的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                word = sentence[i]
                left_contexts = sentence[max(0, i - context_size):i]
                right_contexts = sentence[(i + 1):(
                    min(len(sentence), i + context_size) + 1)]
                # 共现次数随距离衰减: 1/d(w, c)
                for k, contexts_words in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(word, contexts_words)] += 1 / (k + 1)
                for k, contexts_words in enumerate(right_contexts):
                    self.cooccur_counts[(word, contexts_words)] += 1 / (k + 1)
        self.data = [
            (word, contexts_words, count)
            for (word, contexts_words), count in self.cooccur_counts.items()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        return (words, contexts, counts)
