'''
Author: jianzhnie
Date: 2021-12-29 16:07:11
LastEditTime: 2022-01-04 10:37:58
LastEditors: jianzhnie
Description:

'''

import re
from typing import List, Union

import jieba

from .vocab import Vocab


class BaseTokenizer(object):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class Tokenizer:
    """
    Tokenizes text lines into word or character tokens.

    Args:
        lang (str, optional): Language identifier. Default is 'en'.

    Attributes:
        lang (str): Language identifier.

    Usage:
    ```
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Hello, World!", token='word')
    ```

    Defined in :numref:`sec_utils`
    """
    def __init__(self, lang: str = 'en'):
        self.lang = lang

    def tokenize(self,
                 sentence: str,
                 token: str = 'word') -> Union[List[str], List[List[str]]]:
        """
        Tokenize the input sentence into word or character tokens.

        Args:
            sentence (str): The input sentence to tokenize.
            token (str, optional): Token type. Either 'word' or 'char'. Default is 'word'.

        Returns:
            Union[List[str], List[List[str]]]: A list of tokens.

        Raises:
            ValueError: If an unknown token type is provided.

        Usage:
        ```
        tokens = tokenizer.tokenize("Hello, World!", token='word')
        ```

        """
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", ' ',
                          str(sentence))
        sentence = re.sub(r'[ ]+', ' ', sentence)
        sentence = re.sub(r'\!+', '!', sentence)
        sentence = re.sub(r'\,+', ',', sentence)
        sentence = re.sub(r'\?+', '?', sentence)
        sentence = sentence.lower()

        if token == 'word':
            return sentence.split()
        elif token == 'char':
            return [list(word) for word in sentence.split()]
        else:
            raise ValueError('Unknown token type: ' + token)


class JiebaTokenizer(BaseTokenizer):
    """
    Constructs a tokenizer based on `jieba <https://github.com/fxsjy/jieba>`__.
    It supports :meth:`cut` method to split the text to tokens, and :meth:`encode`
    method to covert text to token ids.

    Args:
        vocab(paddlenlp.data.Vocab): An instance of :class:`paddlenlp.data.Vocab`.
    """
    def __init__(self, vocab: Vocab):
        super(JiebaTokenizer, self).__init__(vocab)

        self.tokenizer = jieba.Tokenizer()
        # initialize tokenizer
        self.tokenizer.FREQ = {
            key: 1
            for key in self.vocab.token_to_idx.keys()
        }
        self.tokenizer.total = len(self.tokenizer.FREQ)
        self.tokenizer.initialized = True

    def cut(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to cut the text to tokens.

        Args:
            sentence(str): The text that needs to be cuted.
            cut_all(bool, optional): Whether to use the full mode. If True,
                using full mode that gets all the possible words from the
                sentence, which is fast but not accurate. If False, using
                accurate mode that attempts to cut the sentence into the most
                accurate segmentations, which is suitable for text analysis.
                Default: False.
            use_hmm(bool, optional): Whether to use the HMM model. Default: True.

        Returns:
            list[str]: A list of tokens.

        Example:
            .. code-block:: python

                from paddlenlp.data import Vocab, JiebaTokenizer
                # The vocab file. The sample file can be downloaded firstly.
                # wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
                vocab_file_path = './senta_word_dict.txt'
                # Initialize the Vocab
                vocab = Vocab.load_vocabulary(
                    vocab_file_path,
                    unk_token='[UNK]',
                    pad_token='[PAD]')
                tokenizer = JiebaTokenizer(vocab)

                tokens = tokenizer.cut('我爱你中国')
                print(tokens)
                # ['我爱你', '中国']
        """
        return self.tokenizer.lcut(sentence, cut_all, use_hmm)

    def encode(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to convert the text to ids. It will firstly call
        :meth:`cut` method to cut the text to tokens. Then, convert tokens to
        ids using `vocab`.

        Args:
            sentence(str): The text that needs to be cuted.
            cut_all(bool, optional): Whether to use the full mode. If True,
                using full mode that gets all the possible words from the
                sentence, which is fast but not accurate. If False, using
                accurate mode that attempts to cut the sentence into the most
                accurate segmentations, which is suitable for text analysis.
                Default: False.
            use_hmm(bool, optional): Whether to use the HMM model. Default: True.

        Returns:
            list[int]: A list of ids.

        Example:
            .. code-block:: python

                from paddlenlp.data import Vocab, JiebaTokenizer
                # The vocab file. The sample file can be downloaded firstly.
                # wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
                vocab_file_path = './senta_word_dict.txt'
                # Initialize the Vocab
                vocab = Vocab.load_vocabulary(
                    vocab_file_path,
                    unk_token='[UNK]',
                    pad_token='[PAD]')
                tokenizer = JiebaTokenizer(vocab)

                ids = tokenizer.encode('我爱你中国')
                print(ids)
                # [1170578, 575565]
        """
        words = self.cut(sentence, cut_all, use_hmm)

        return [
            self.get_idx_from_word(word, self.vocab.token_to_idx,
                                   self.vocab.unk_token) for word in words
        ]

    def get_idx_from_word(self, word, word_to_idx, unk_word):
        if word in word_to_idx:
            return word_to_idx[word]
        return word_to_idx[unk_word]
