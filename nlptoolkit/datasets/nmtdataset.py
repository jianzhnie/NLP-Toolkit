import sys

import torch

sys.path.append('../../')
from typing import List, Tuple

from nlptoolkit.data.tokenizer import Tokenizer
from nlptoolkit.data.utils.utils import truncate_pad
from nlptoolkit.data.vocab import (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
                                   Vocab)


class NMTDatasets():
    """Machine translation dataset.

    Args:
        file_path (str): The path to the  machine translation data file.

    """
    def __init__(self, file_path: str = 'data', max_seq_len: int = 10):
        super(NMTDatasets, self).__init__()
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()
        self.src_txt, self.tgt_tx = self.load_and_preprocess_data(
            self.file_path)
        self.src_sentences, self.tgt_sentences = self._tokenize(
            self.src_txt, self.tgt_tx)
        self.src_vocab = Vocab.build_vocab(self.src_sentences,
                                           min_freq=2,
                                           pad_token=PAD_TOKEN,
                                           unk_token=UNK_TOKEN,
                                           bos_token=BOS_TOKEN,
                                           eos_token=EOS_TOKEN)
        self.tgt_vocab = Vocab.build_vocab(self.tgt_sentences,
                                           min_freq=2,
                                           pad_token=PAD_TOKEN,
                                           unk_token=UNK_TOKEN,
                                           bos_token=BOS_TOKEN,
                                           eos_token=EOS_TOKEN)

    def load_and_preprocess_data(
            self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load and preprocess the machine translation dataset.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: A tuple containing two lists.
                - The first list contains source sentences.
                - The second list contains target sentences.
        """
        source_texts = []
        target_texts = []
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\u202f', ' ').replace('\xa0', ' ')
                parts = line.split('\t')
                if len(parts):
                    source_texts.append(parts[0])
                    target_texts.append(parts[1])
        assert len(source_texts) == len(
            target_texts
        ), 'The number of source sentences and target sentences does not match.'

        return source_texts, target_texts

    def _tokenize(self, src_texts, tgt_texts):
        """Tokenize the  dataset."""
        src_sentences = []
        tgt_sentences = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_tokens = self.tokenizer.tokenize(src_text)
            tgt_tokens = self.tokenizer.tokenize(tgt_text)
            src_sentences.append(src_tokens)
            tgt_sentences.append(tgt_tokens)
        assert len(src_sentences) == len(
            tgt_sentences
        ), 'The number of source sentences and target sentences does not match.'

        return src_sentences, tgt_sentences

    def __getitem__(self, idx):
        src_txt = self.src_sentences[idx]
        tgt_txt = self.tgt_sentences[idx]
        src_tokens = self.src_vocab.to_index(src_txt)
        src_tokens = truncate_pad(src_tokens, self.max_seq_len,
                                  self.src_vocab['<pad>'])
        tgt_tokens = self.tgt_vocab.to_index(tgt_txt)
        tgt_tokens = truncate_pad(tgt_tokens, self.max_seq_len,
                                  self.tgt_vocab['<pad>'])
        src_tensor, tgt_tensor = self._to_tensor(src_tokens, tgt_tokens)
        return src_tensor, tgt_tensor

    def __len__(self):
        return len(self.src_txt)

    def _to_tensor(self, src_tokens, tgt_tokens):
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        return src_tensor, tgt_tensor


if __name__ == '__main__':
    root = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    nmtdataset = NMTDatasets(file_path=root)
    for idx in range(10):
        src_tensor, tgt_tensor = nmtdataset[idx]
        print(src_tensor, tgt_tensor)
