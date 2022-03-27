import os
import sys

import d2l
import torch
from torch.utils import data

from nlptoolkit.data.vocab import Vocab

sys.path.append('../../')


class NMTDatasets():
    """Defined in :numref:`sec_machine_translation`"""
    def __init__(self,
                 root='../data',
                 num_steps=10,
                 num_train=1000,
                 num_val=1000,
                 download=True):
        super(NMTDatasets, self).__init__()
        self.root = root
        self.download = download

        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._read_data())

    def _read_data(self):
        """Load the English-French dataset."""
        if self.download:
            d2l.extract(
                d2l.download(d2l.DATA_URL + 'fra-eng.zip', self.root,
                             '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        """Preprocess the English-French dataset."""

        # Insert space between words and punctuation marks
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        out = [
            ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text.lower())
        ]
        return ''.join(out)

    def _tokenize(self, text, max_examples=None):
        """Tokenize the English-French dataset."""
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append(parts[0].split(' '))
                tgt.append(parts[1].split(' '))
        return src, tgt

    def _truncate_pad(self, line, num_steps, padding_token):
        """Truncate or pad sequences."""

        if len(line) > num_steps:
            return line[:num_steps]  # Truncate
        return line + [padding_token] * (num_steps - len(line))  # Pad

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        """Defined in :numref:`sec_machine_translation`"""
        def _build_one(sentences, vocab):
            sentences = [
                self._truncate_pad(seq, self.num_steps, ['<pad>'])
                for seq in sentences
            ]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)
            array = d2l.tensor([vocab[sent] for sent in sentences])
            return array, vocab

        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab = _build_one(src, src_vocab)
        tgt_array, tgt_vocab = _build_one(tgt, tgt_vocab)
        return (src_array, tgt_array[:, :-1],
                tgt_array[:, 1:]), src_vocab, tgt_vocab

    def build(self, src_sentences, tgt_sentences):
        """Defined in :numref:`sec_machine_translation`"""
        raw_text = '\n'.join([
            src + '\t' + tgt for src, tgt in zip(src_sentences, tgt_sentences)
        ])
        arrays, _, _ = self._build_arrays(raw_text, self.src_vocab,
                                          self.tgt_vocab)
        return arrays

    def get_dataloader(self, train):
        """Defined in :numref:`sec_machine_translation`"""
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def get_train_dataloader(self):
        return self.get_dataloader(train=True)

    def get_val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset,
                                           self.batch_size,
                                           shuffle=train)


def read_data_nmt():
    """Load the English-French dataset.

    Defined in :numref:`sec_utils`
    """
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_utils`
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset.

    Defined in :numref:`sec_utils`
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def preprocess_nmt(text):
    """Preprocess the English-French dataset.

    Defined in :numref:`sec_utils`
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return ''.join(out)


def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches.

    Defined in :numref:`sec_utils`
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(d2l.astype(array != vocab['<pad>'], d2l.int32),
                               1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset.

    Defined in :numref:`sec_utils`
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source,
                          min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target,
                          min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


if __name__ == '__mian__':

    nmtdataset = NMTDatasets(download=True)
    print(nmtdataset)
    data1 = nmtdataset.arrays
    print(data1)
