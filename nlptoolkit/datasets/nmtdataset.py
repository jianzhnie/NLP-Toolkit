import torch

from nlptoolkit.data.vocab import Vocab


class NMTDatasets():
    """Defined in :numref:`sec_machine_translation`"""
    def __init__(self,
                 root='data',
                 max_seq_len=10,
                 num_train=1000,
                 num_val=1000):
        super(NMTDatasets, self).__init__()
        self.root = root
        self.max_seq_len = max_seq_len
        self.num_train = num_train
        self.num_val = num_val

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

    def _truncate_pad(self, line, max_seq_len, padding_token):
        """Truncate or pad sequences."""

        if len(line) > max_seq_len:
            return line[:max_seq_len]  # Truncate
        return line + [padding_token] * (max_seq_len - len(line))  # Pad

    def _bulid_array_nmt(self, tokens, vocab=None, max_seq_len=10):
        """Transform text sequences of machine translation into minibatches."""
        if vocab is None:
            vocab = Vocab.build_vocab(tokens, min_freq=2)

        text_tokens = [vocab[token] for token in tokens]
        text_tokens = [[vocab['<bos>']] + token + [vocab['<eos>']]
                       for token in text_tokens]
        text_tokens = [
            self._truncate_pad(l, max_seq_len, vocab['<pad>'])
            for l in text_tokens
        ]
        return text_tokens, vocab

    def get_dataset_tokens(self,
                           raw_text=None,
                           src_vocab=None,
                           tgt_vocab=None):
        """Defined in :numref:`sec_machine_translation`"""
        if raw_text is None:
            raw_text = self._read_data()

        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_tokens, src_vocab = self._bulid_array_nmt(src, src_vocab)
        tgt_tokens, tgt_vocab = self._bulid_array_nmt(tgt, tgt_vocab)
        return src_tokens, tgt_tokens, src_vocab, tgt_vocab

    def get_tensor_dataset(self, src_tokens, tgt_tokens, train=True):
        indices = slice(0, self.num_train) if train else slice(
            self.num_train, None)

        src_tokens, tgt_tokens = tuple(datasets[indices]
                                       for datasets in (src_tokens,
                                                        tgt_tokens))
        data = []
        for (src_token, tgt_token) in zip(src_tokens, tgt_tokens):
            src_tensor_ = torch.tensor([token for token in src_token],
                                       dtype=torch.long)
            tgt_tensor_ = torch.tensor([token for token in tgt_token],
                                       dtype=torch.long)
            data.append((src_tensor_, tgt_tensor_))
        return data


if __name__ == '__main__':
    root = 'data'
    nmtdataset = NMTDatasets(root=root)
    print(nmtdataset)
    # arrays, src_vocab, tgt_vocab = nmtdataset._build_arrays(
    #     nmtdataset._read_data())
    data1 = nmtdataset._read_data()
    data2 = nmtdataset._preprocess(data1)
    src, tgt = nmtdataset._tokenize(data2)
    # print(src, tgt)
    src_tokens, tgt_tokens, src_vocab, tgt_vocab = nmtdataset._build_tokens(
        data1)
    data_train = nmtdataset.get_tensor_dataset(src_tokens, tgt_tokens)
    print(src_tokens[0], tgt_tokens[0])
    print(data_train[0])
