'''
Author: jianzhnie
Date: 2021-12-17 12:13:36
LastEditTime: 2022-01-05 15:25:13
LastEditors: jianzhnie
Description:

'''
import random
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
BOW_TOKEN = '<bow>'
EOW_TOKEN = '<eow>'


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


def generate_ngram_dataset(sentence, context_size):
    data = []
    for i in range(context_size, len(sentence)):
        context = [sentence[i - j - 1] for j in range(context_size)]
        context = context[::-1]
        target = sentence[i]
        data.append((context, target))
    return data


def generate_cbow_dataset(sentence, context_size):
    data = []
    for i in range(context_size, len(sentence) - context_size):
        context_befor = [sentence[i - j - 1] for j in range(context_size)]
        context_befor = context_befor[::-1]
        context_after = [sentence[i + j + 1] for j in range(context_size)]
        target = sentence[i]
        data.append((context_befor + context_after, target))
    return data


def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词."""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(
                range(max(0, i - window_size),
                      min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


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
    CONTEXT_SIZE = 4  # 2 words to the left, 2 to the right
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    ngram_data = generate_ngram_dataset(raw_text, CONTEXT_SIZE)
    print(ngram_data)

    cbow_data = generate_cbow_dataset(raw_text, CONTEXT_SIZE)
    print(cbow_data)

    centers, contexts = get_centers_and_contexts(raw_text, 2)

    train_data, test_data, vocab = load_sentence_polarity()
    print(train_data[:10])

    train_data, test_data, vocab, tag_vocab = load_treebank()
    print(train_data[:10])
    print(test_data[:10])
