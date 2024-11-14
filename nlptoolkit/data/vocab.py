"""A comprehensive vocabulary management system for NLP tasks.

This module provides a Vocab class that handles token-to-index mapping, special
tokens, and various utilities for text processing tasks.
"""

import collections
import io
import json
import os
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Union

UNK_TOKEN = '<UNK>'
SEP_TOKEN = '<SEP>'
PAD_TOKEN = '<PAD>'
CLS_TOKEN = '<CLS>'
MASK_TOKEN = '<MASK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
BOW_TOKEN = '<BOW>'
EOW_TOKEN = '<EOW>'


def tokenize(lines: List[str], token: str = "word") -> List[List[str]]:
    """Split text lines into word or character tokens.

    Args:
        lines: List of input text lines to tokenize
        token: Tokenization type, either 'word' or 'char'

    Returns:
        List of tokenized sequences

    Raises:
        AssertionError: If token type is invalid
    """
    assert token in ("word", "char"), f"Unknown token type: {token}"
    return [line.split() if token == 'word' else list(line) for line in lines]


def count_corpus(tokens: Union[List[str], List[List[str]]]) -> Counter:
    """Count token frequencies in a corpus.

    Args:
        tokens: Input tokens, either as a flat list or list of sequences

    Returns:
        Counter object with token frequencies
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten 2D list of tokens into 1D
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)


class Vocab(object):
    """Vocabulary for text.

    The class is used to convert between tokens and ids. It also includes some
    store/load functions.

    Args:
        counter (collections.Counter, optional): A Counter instance describing
            the tokens and their frequencies. Its keys will be indexed according
            to the order of frequency sorting to construct the mapping relationship.
            If None, `token_to_idx` must be provided as the mapping relationship.
            Default: None.
        max_size (int, optional): Max size of vocab, not including special tokens.
            Default: None.
        min_freq (int, optional): Ignore tokens whose frequencies are less than
            `min_freq`. Default: 1.
        token_to_idx (dict, optional): A dict specifying the mapping relationship
            between tokens and indices to be used. If provided, adjust the tokens
            and indices mapping according to it. If None, counter must be provided.
            Default: None.
        unk_token (str, optional): Special token for unknown token. If not needed,
            it can be None. Default: '<unk>'.
        pad_token (str, optional): Special token for padding token. If not needed,
            it can be None. Default: '<pad>'.
        bos_token (str, optional): Special token for beginning of sentence (BOS) token.
            If not needed, it can be None. Default: '<bos>'.
        eos_token (str, optional): Special token for end of sentence (EOS) token.
            If not needed, it can be None. Default: '<eos>'.
        kwargs (dict): Keyword arguments ending with '_token'. It can be used
            to specify further special tokens that will be exposed as attributes
            of the vocabulary and associated with an index.
    """

    def __init__(
        self,
        counter: Optional[Counter] = None,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        token_to_idx: Optional[Dict[str, int]] = None,
        unk_token: Optional[str] = UNK_TOKEN,
        pad_token: Optional[str] = PAD_TOKEN,
        bos_token: Optional[str] = BOS_TOKEN,
        eos_token: Optional[str] = EOS_TOKEN,
        **kwargs: Any,
    ) -> None:
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Handle special tokens
        special_token_dict = OrderedDict({
            'pad_token': pad_token,
            'unk_token': unk_token,
            'bos_token': bos_token,
            'eos_token': eos_token,
        })
        special_token_dict.update(kwargs)

        # Check if special tokens are in kwargs
        special_tokens: List[str] = []
        self.special_token_dict = OrderedDict()
        for token_name, special_token in special_token_dict.items():
            # Test if kwarg specifies a special token
            if not token_name.endswith("_token"):
                raise ValueError(
                    f"{token_name} is invalid. Only keyword arguments "
                    "that end in '_token' are supported "
                    "to declare special tokens."
                )

            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)
                self.special_token_dict[token_name] = special_token

        # Map tokens to indices and indices to tokens
        if counter is None:
            assert (
                token_to_idx is not None
            ), "token_to_idx must be provided if counter is None"
            self.token_to_idx: Dict[str, int] = token_to_idx
            self.idx_to_token: Dict[int, str] = {
                idx: token for token, idx in token_to_idx.items()
            }
            self.add_tokens(special_tokens)

        else:
            # Map special tokens to index
            self.idx_to_token = {idx: token for idx, token in enumerate(special_tokens)}
            self.token_to_idx = OrderedDict()
            self.token_to_idx.update(
                (token, idx) for idx, token in self.idx_to_token.items()
            )

            self._index_counter_keys(counter, special_tokens, max_size, min_freq)

    def _index_counter_keys(
        self,
        counter: Counter,
        special_tokens: List[str],
        max_size: Optional[int],
        min_freq: int,
    ) -> None:
        """Index tokens from counter while respecting frequency and size
        constraints.

        Args:
            counter: Token counter
            special_tokens: List of special tokens to exclude from indexing
            max_size: Maximum vocabulary size
            min_freq: Minimum frequency threshold
        """
        # Sort by frequency, then alphabetically
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.word_counter = {key: val for key, val in self.token_freqs}
        # Frequencies of special tokens are not counted when building vocabulary
        # in frequency order

        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)

        for token, freq in self.token_freqs:
            if freq < min_freq or len(self.idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self.add_tokens(token)

    def add_tokens(self, tokens: Union[str, List[str]]) -> None:
        """Add new tokens to the vocabulary.

        Args:
            tokens: Single token or list of tokens to add
        """
        if not isinstance(tokens, (list, tuple)):
            if tokens not in self.token_to_idx:
                idx = len(self.idx_to_token)
                self.idx_to_token[idx] = tokens
                self.token_to_idx[tokens] = idx
        else:
            for token in tokens:
                self.add_tokens(token)

    def to_tokens(self, indices: List[int]) -> Union[List[str], str]:
        """Maps the input indices to a token or list of tokens.

        Args:
            indices (int|list[int]|tuple[int]|numpy.ndarray): The input indice(s) for mapping.
                Must be an `int` or 1D `list[int]`|`tuple[int]`|`numpy.ndarray`.

        Returns:
            str|list[str]: Obtained token(s). If `indices` is an integer, it
            will return a str. If `indices` is a list/tuple of integers, it will
            return a list of str.
        """
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    def to_index(self, tokens: Union[str, List[str]]):
        """Maps the input tokens into indices.

        Args:
            tokens (str|list[str]|tuple[str], optional): The input token(s) for
                mapping.

        Returns:
            int|list[int]: Obtained indice(s). If `tokens` is a str, it will
            return an integer. If `tokens` is a list/tuple of str, it will
            return a list of integers.
        """
        return self[tokens]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def size(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx.get(self.unk_token))
        return [self.__getitem__(token) for token in tokens]

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_idx

    def get_bos_token_id(self) -> Optional[int]:
        """Get the index of the beginning-of-sequence token."""
        return (
            self.token_to_idx.get(self.bos_token)
            if self.bos_token is not None
            else None
        )

    def get_eos_token_id(self) -> Optional[int]:
        """Get the index of the end-of-sequence token."""
        return (
            self.token_to_idx.get(self.eos_token)
            if self.eos_token is not None
            else None
        )

    def get_pad_token_id(self) -> Optional[int]:
        """Get the index of the padding token."""
        return (
            self.token_to_idx.get(self.pad_token)
            if self.pad_token is not None
            else None
        )

    def get_token_freq(self):
        return self.word_counter

    def to_json(self, path: str = None):
        """Summarizes some information of the vocab as a JSON string. If a path
        is.

        # JSON string. The JSON string and the saved file can both be used to reconstruct the `Vocab`
        # by calling the `from_json` method.

        Args:
            path (str, optional): The path to save the JSON string. If None, the JSON will not be saved.
                Default: None.

        Returns:
            str: The JSON string including information of vocab.
        """
        vocab_dict = {}
        vocab_dict["idx_to_token"] = dict(self.idx_to_token)
        vocab_dict["token_to_idx"] = dict(self.token_to_idx)
        vocab_dict["special_token"] = self.special_token_dict
        vocab_dict["token_freqs"] = self.token_freqs
        json_str = json.dumps(vocab_dict, indent=2, ensure_ascii=False)
        if path:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str: str):
        """Loads a `Vocab` from a JSON string or JSON file, which is generated
        by calling the `to_json` method.

        Args:
            json_str (str): JSON string or file path to a JSON string.

        Returns:
            Vocab: An instance of `Vocab` generated from the information contained in the JSON string.
        """
        if os.path.isfile(json_str):
            with io.open(json_str, "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get("token_to_idx")
        special_tokens = vocab_dict.get("special_token", dict())
        vocab = cls(counter=None, token_to_idx=token_to_idx, **special_tokens)
        return vocab

    @classmethod
    def from_dict(
        cls,
        token_to_idx: Dict[str, int],
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Builds the `Vocab` from a dictionary.

        Args:
            token_to_idx (dict): A dictionary describing the mapping relationship between tokens and indices.
            unk_token (str, optional): The special token for unknown tokens '<unk>'. If not needed, it can
                be None. Default: None.
            pad_token (str, optional): The special token for padding tokens '<pad>'. If not needed, it can
                be None. Default: None.
            bos_token (str, optional): The special token for beginning of sentence (BOS) token '<bos>'.
                If not needed, it can be None. Default: None.
            eos_token (str, optional): The special token for end of sentence (EOS) token '<eos>'.
                If not needed, it can be None. Default: None.
            kwargs (dict): Keyword arguments ending with '_token'. It can be used to specify further special
                tokens that will be exposed as attributes of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of `Vocab` generated from the given dictionary and special tokens.
        """
        vocab = cls(
            counter=None,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        return vocab

    @staticmethod
    def build_vocab(
        iterator: Iterable,
        max_size: int = None,
        min_freq: int = 1,
        token_to_idx: Dict[str, int] = None,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Builds the `Vocab` according to a given iterator and other
        information. It first iterates over the `iterator` to construct a
        `collections.Counter` and uses it to initialize the `Vocab`.

        Args:
            iterator (collections.Iterable): Iterator of tokens. Each element should be a list of tokens if
                word-level vocab is needed.
            max_size (int, optional): The max size of vocab, not including special tokens. Default: None.
            min_freq (int, optional): Ignore tokens whose frequencies are less than `min_freq`. Default: 1.
            token_to_idx (dict, optional): A dict specifying the mapping relationship between tokens and
                indices to be used. If provided, adjust the tokens and indices mapping according to it.
                If None, counter must be provided. Default: None.
            unk_token (str, optional): The special token for unknown tokens '<unk>'. If not needed, it can
                be None. Default: None.
            pad_token (str, optional): The special token for padding tokens '<pad>'. If not needed, it can
                be None. Default: None.
            bos_token (str, optional): The special token for beginning of sentence (BOS) token '<bos>'.
                If not needed, it can be None. Default: None.
            eos_token (str, optional): The special token for end of sentence (EOS) token '<eos>'.
                If not needed, it can be None. Default: None.
            kwargs (dict): Keyword arguments ending with '_token'. It can be used to specify further special
                tokens that will be exposed as attributes of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of `Vocab` generated from the given iterator and other information.
        """
        if iterator and isinstance(iterator[0], (list, tuple)):
            iterator = [token for line in iterator for token in line]
        counter = collections.Counter(iterator)
        vocab = Vocab(
            counter,
            max_size=max_size,
            min_freq=min_freq,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        return vocab

    def save_vocabulary(self, path: str):
        """Save the vocabulary to a file."""
        with open(path, "w") as writer:
            for idx in range(len(self.idx_to_token)):
                writer.write(self.idx_to_token[idx] + "\n")

    @classmethod
    def load_vocabulary(
        cls,
        vocab_file: str,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        **kwargs,
    ):
        """Builds the `Vocab` from a file, preserving all tokens by calling the
        `from_dict` method. The file contains a token per line, and the line
        index would be the index of the corresponding token.

        Args:
            vocab_file (str): The path of the file used to construct the vocabulary.
            unk_token (str, optional): The special token for unknown tokens '<unk>'. If not needed, it can
                be None. Default: None.
            pad_token (str, optional): The special token for padding tokens '<pad>'. If not needed, it can
                be None. Default: None.
            bos_token (str, optional): The special token for beginning of sentence (BOS) token '<bos>'.
                If not needed, it can be None. Default: None.
            eos_token (str, optional): The special token for end of sentence (EOS) token '<eos>'.
                If not needed, it can be None. Default: None.
            kwargs (dict): Keyword arguments ending with '_token'. It can be used to specify further special
                tokens that will be exposed as attributes of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of `Vocab` generated from the given file.
        """

        token_lst = []
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
            for index, token in enumerate(tokens):
                token = token.rstrip("\n").split("\t")[0]
                vocab[token] = index
        token_lst = list(vocab.keys())
        token_to_idx = {token: idx for idx, token in enumerate(token_lst)}
        vocab = cls.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        return vocab
