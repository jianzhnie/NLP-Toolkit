'''
Author: jianzhnie
Date: 2022-01-05 09:40:22
LastEditTime: 2022-03-04 17:17:59
LastEditors: jianzhnie
Description:

'''
import collections
import math
import random
import sys
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

sys.path.append('../../')
from nlptoolkit.utils.data_utils import (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
                                         UNK_TOKEN)


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


class WordSubsampler:
    def __init__(self,
                 raw_texts: List[List[str]],
                 unk_token: str = UNK_TOKEN,
                 threshold: float = 1e-4):
        """
        Initialize the WordSubsampler.

        Args:
            raw_texts (List[List[str]]): A list of lists containing tokenized sentences.
            unk_token (str): The token representing unknown words.
            threshold (float): The subsampling threshold.
        """
        self.raw_texts = raw_texts
        self.unk_token = unk_token
        self.threshold = threshold
        self.counter = None

    def exclude_unknown_tokens(self, raw_texts) -> List[List[str]]:
        """
        Exclude unknown tokens ('<unk>') from the raw texts.
        Args:
            raw_texts (List[List[str]]): List of sentences with unknown tokens removed.

        Returns:
            List[List[str]]: List of sentences with unknown tokens removed.
        """
        return [[token for token in line if token != self.unk_token]
                for line in raw_texts]

    def calculate_word_frequencies(
            self, raw_texts: List[List[str]]) -> collections.Counter:
        """
        Calculate word frequencies from the raw texts.

        Args:
            raw_texts (List[List[str]]): List of sentences with unknown tokens removed.

        Returns:
            collections.Counter: Counter object with word frequencies.
        """
        return collections.Counter(
            [token for line in raw_texts for token in line])

    def keep(self, token: str, counter: collections.Counter) -> bool:
        """
        Determine whether to keep a token based on word frequencies and threshold.

        Args:
            token (str): The token to evaluate.
            counter (collections.Counter): Counter object with word frequencies.

        Returns:
            bool: True if the token should be kept, False otherwise.
        """
        if counter is None:
            raise ValueError(
                'Word frequencies not calculated. Call calculate_word_frequencies() first.'
            )
        num_tokens = sum(counter.values())
        ratio = counter[token] / num_tokens
        keep_flag = (random.uniform(0, 1) < math.sqrt(self.threshold / ratio))
        return keep_flag

    def subsample_words(self) -> List[List[str]]:
        """
        Perform word subsampling based on word frequencies and threshold.

        Returns:
            List[List[str]]: List of sentences with subsampled words.
        """
        self.sentences = self.exclude_unknown_tokens(self.raw_texts)
        self.counter = self.calculate_word_frequencies(self.sentences)
        self.subsampled_words = [[
            token for token in line if self.keep(token, self.counter)
        ] for line in self.raw_texts]
        return self.subsampled_words

    def get_word_frequencies(self, token: str):
        """
        Get the frequency of a specific token.

        Args:
            token (str): The token to query.

        Returns:
            int: The frequency count of the token.
        """
        if self.counter is None:
            raise ValueError(
                'Word frequencies not calculated. Call calculate_word_frequencies() first.'
            )
        origin_counts = sum(
            [sentence.count(token) for sentence in self.sentences])
        subsampled_counts = sum(
            [sentence.count(token) for sentence in self.subsampled_words])

        return origin_counts, subsampled_counts

    def get_word_counter(self):
        if self.counter is None:
            raise ValueError(
                'Word frequencies not calculated. Call calculate_word_frequencies() first.'
            )
        return self.counter


def generate_ngram_dataset(sentence: List[str],
                           context_size: int) -> List[Tuple[List[str], str]]:
    """
    Generate an n-gram dataset from a given sentence.

    Args:
        sentence (List[str]): The input sentence as a list of words.
        context_size (int): The size of the context for each n-gram.

    Returns:
        List[Tuple[List[str], str]]: A list of tuples, each containing a context list and a target word.

    Example:
        >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
        >>> context_size = 2
        >>> data = generate_ngram_dataset(sentence, context_size)
        >>> print(data)
        [(['I', 'love'], 'to'), (['love', 'to'], 'eat'), (['to', 'eat'], 'ice'), (['eat', 'ice'], 'cream')]
    """
    ngram_data = []
    sentence_length = len(sentence)
    if sentence_length < context_size:
        return []

    for i in range(context_size, sentence_length):
        # Construct the context for the current n-gram
        context = sentence[i - context_size:i]
        target = sentence[i]
        ngram_data.append((context, target))

    return ngram_data


def generate_cbow_dataset(sentence: List[str],
                          context_size: int) -> List[Tuple[List[str], str]]:
    """
    Generate a CBOW (Continuous Bag of Words) dataset from a given sentence.

    Args:
        sentence (List[str]): The input sentence as a list of words.
        context_size (int): The context window size, determining the number of words to consider before and after the target word.

    Returns:
        List[Tuple[List[str], str]]: A list of tuples, where each tuple contains a list of context words and the target word.

    Example:
        >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
        >>> context_size = 2
        >>> data = generate_cbow_dataset(sentence, context_size)
        >>> print(data)
        [(['love', 'to'], 'I'),
        (['I', 'to', 'eat'], 'love'),
        (['I', 'love', 'eat', 'ice'], 'to'),
        (['love', 'to', 'ice', 'cream'], 'eat'),
        (['to', 'eat', 'cream'], 'ice'),
        (['eat', 'ice'], 'cream')]
    """
    cbow_data = []
    sentence_length = len(sentence)
    if sentence_length < 2:
        return []
    for i in range(sentence_length):
        # Determine the context word indices within the window
        left_idx = max(0, i - context_size)
        right_idx = min(sentence_length, i + 1 + context_size)
        context_indices = list(range(left_idx, right_idx))
        # Get the target word
        center_words = sentence[i]
        # Exclude the target word from the context words
        context_indices.remove(i)
        context_words = [sentence[idx] for idx in context_indices]

        # Append the context and target as a tuple to the dataset
        cbow_data.append((context_words, center_words))

    return cbow_data


def generate_skipgram_dataset(
        sentence: List[str],
        context_size: int) -> Tuple[List[str], List[List[str]]]:
    """
    Generate an Skip-gram dataset from a given sentence.
    返回跳元模型中的中心词和上下文词.

    Args:
        sentence (List[str]): The input sentence as a list of words.
        context_size (int): The maximum window size for context words.

    Returns:
        List[Tuple[List[str], str]]: A list of tuples, where each tuple contains a list of context words and the target word.

    Example:
    ```python
        >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
        >>> context_size = 2
        >>> data = generate_skipgram_dataset(sentence, context_size)
        >>> print(data)
        [(['love', 'to'], 'I'),
        (['I', 'to', 'eat'], 'love'),
        (['I', 'love', 'eat', 'ice'], 'to'),
        (['love', 'to', 'ice', 'cream'], 'eat'),
        (['to', 'eat', 'cream'], 'ice'),
        (['eat', 'ice'], 'cream')]
    ```
    """
    skipgram_data = []

    sentence_length = len(sentence)
    # To form "center word - context word" pairs, a sentence should have at least 2 words.
    if sentence_length < 2:
        return []

    for i in range(sentence_length):
        # Determine the context word indices within the window
        left_idx = max(0, i - context_size)
        right_idx = min(sentence_length, i + 1 + context_size)
        context_indices = list(range(left_idx, right_idx))
        center_word = sentence[i]
        # Exclude the center word from the context words
        context_indices.remove(i)
        context_words = [sentence[idx] for idx in context_indices]
        # Append the center word and context words as a tuple to the dataset
        skipgram_data.append((center_word, context_words))

    return skipgram_data


class RandomSampler:
    """
    Randomly sample integers from a population based on given weights.

    This class allows you to sample integers from 1 to n based on provided sampling weights.

    Args:
        sampling_weights (List[float]): A list of sampling weights for each integer in the population.


    Example:
        >>> sampler = RandomSampler([0.2, 0.3, 0.5])
        >>> sample = sampler.draw()
    """
    def __init__(self, sampling_weights: List[float]):
        """
        Initialize the RandomSampler with sampling weights.

        Args:
            sampling_weights (List[float]): A list of sampling weights for each integer in the population.
        """
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self) -> int:
        """
        Draw a random sample based on the provided weights.

        Returns:
            int: A randomly sampled integer.
        """
        if self.i == len(self.candidates):
            # Cache k random samples
            self.candidates = random.choices(self.population,
                                             self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


# Example usage:
if __name__ == '__main__':
    sampler = RandomSampler([0.2, 0.3, 0.5])
    sample = sampler.draw()
    print(f'Randomly sampled value: {sample}')

    corpus = ['I', 'love', 'to', 'eat', 'ice', 'cream']
    print(corpus)
    context_size = 2

    ngrm = generate_ngram_dataset(corpus, context_size)
    print(ngrm)

    cbow = generate_cbow_dataset(corpus, context_size)
    print(cbow)

    skip = generate_skipgram_dataset(corpus, context_size)
    print(skip)
