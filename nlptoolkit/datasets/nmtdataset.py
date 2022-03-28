import sys

import torch

from nlptoolkit.data.vocab import Vocab

sys.path.append('../../')


class NMTDatasets():
    """Defined in :numref:`sec_machine_translation`"""
    def __init__(self,
                 root='../data',
                 batch_size=32,
                 num_steps=10,
                 num_train=1000,
                 num_val=1000):
        super(NMTDatasets, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val

        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._read_data())

    def _read_data(self):
        """Load the English-French dataset."""
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

    def _bulid_array_nmt(self, tokens, vocab=None, num_steps=10):
        """Transform text sequences of machine translation into minibatches."""
        if vocab is None:
            vocab = Vocab(tokens,
                          min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

        text_tokens = [vocab[token] for token in tokens]
        text_tokens = [token + [vocab['<eos>']] for token in text_tokens]
        data_array = torch.tensor([
            self.truncate_pad(l, num_steps, vocab['<pad>'])
            for l in text_tokens
        ])
        return data_array, vocab

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        """Defined in :numref:`sec_machine_translation`"""
        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab = self._bulid_array_nmt(src, src_vocab)
        tgt_array, tgt_vocab = self._bulid_array_nmt(tgt, tgt_vocab)
        return (src_array, tgt_array), src_vocab, tgt_vocab

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


if __name__ == '__main__':
    root = '/Users/jianzhengnie/work_dir/code_gallery/nlp-toolkit/examples/data'
    nmtdataset = NMTDatasets(root=root)
    print(nmtdataset)
    # arrays, src_vocab, tgt_vocab = nmtdataset._build_arrays(
    #     nmtdataset._read_data())
    data1 = nmtdataset._read_data()
    data2 = nmtdataset._preprocess(data1)
    src, tgt = nmtdataset._tokenize(data2)
    # print(src, tgt)
    train_array = nmtdataset.get_dataloader(train=True)
    test_array = nmtdataset.get_dataloader(train=False)
    print(train_array)
    print(test_array)
