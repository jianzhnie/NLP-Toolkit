'''
Author: jianzhnie
Date: 2021-11-30 10:09:54
LastEditTime: 2022-01-04 17:39:48
LastEditors: jianzhnie
Description:

'''
import collections
import io
import json
import os

BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_utils`
    """
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


def count_corpus(tokens):
    """统计词元的频率。

    Defined in :numref:`sec_text_preprocessing`
    """
    # 这里的 `tokens` 是 1D 列表或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成使用词元填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


class Vocab(object):
    """Vocabulary for text.

    The class used to convert between tokens and ids. It also includes some
    store/load functions.

    Args:
        counter (collections.Counter, optional): A Counter intance describes
            the tokens and their frequencies. Its keys will be indexed accroding
            to the order of frequency sorting to construct mapping relationship.
            If None, `token_to_idx` must be provided as the mapping relationship.
            Default: None.
        max_size (int, optional): Max size of vocab, not including special tokens.
            Default: None.
        min_freq (int, optional): Ignore tokens whose frequencies are less than
            `min_freq`. Default: 1.
        token_to_idx (dict, optional): A dict specifies the mapping relationship
            between tokens and indices to be used. If provided, adjust the tokens
            and indices mapping according to it. If None, counter must be provided.
            Default: None.
        unk_token (str, optional): Special token for unknow token. If no need,
            it also could be None. Default: None.
        pad_token (str, optional): Special token for padding token. If no need,
            it also could be None. Default: None.
        bos_token (str, optional): Special token for bos token. If no need, it
            also could be None. Default: None.
        eos_token (str, optional): Special token for eos token. If no need, it
            lso could be None. Default: None.

        kwargs (dict): Keyword arguments ending with `_token`. It can be used
            to specify further special tokens that will be exposed as attribute
            of the vocabulary and associated with an index.
    """
    def __init__(self,
                 counter=None,
                 max_size=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token=UNK_TOKEN,
                 pad_token=PAD_TOKEN,
                 bos_token=BOS_TOKEN,
                 eos_token=EOS_TOKEN,
                 **kwargs):

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Handle special tokens
        self.special_token_dict = {
            'unk_token': unk_token,
            'pad_token': pad_token,
            'bos_token': bos_token,
            'eos_token': eos_token,
        }
        kwargs.update(self.special_token_dict)

        # check if special tokens are in kwargs
        special_tokens = []
        special_iter = kwargs.keys()
        for special_token_name in special_iter:
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError(
                    '{} is invalid. Only keyword arguments '
                    "that end in '_token' are supported "
                    'to declare special tokens.'.format(special_token_name))

            special_token = kwargs[special_token_name]
            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        # map tokens to indices and indices to tokens
        if counter is None:
            assert token_to_idx is not None, 'token_to_idx must be provided if counter is None'
            for special_token in special_tokens:
                assert special_token in token_to_idx, '{} is not in token_to_idx'.format(
                    special_token)

            self._token_to_idx = token_to_idx
            self._idx_to_token = {
                idx: token
                for token, idx in token_to_idx.items()
            }
            if unk_token:
                unk_index = self._token_to_idx[unk_token]
                self._token_to_idx = collections.OrderedDict(lambda: unk_index)
                self._token_to_idx.update(token_to_idx)
        else:
            # map special tokens to index
            self._idx_to_token = {
                idx: special_token
                for idx, special_token in enumerate(special_tokens)
            }
            self._token_to_idx = collections.OrderedDict()
            self._token_to_idx.update(
                (token, idx) for idx, token in self._idx_to_token.items())

            self._index_counter_keys(counter, special_tokens, max_size,
                                     min_freq)

            if token_to_idx:
                self._sort_index_according_to_user_specification(token_to_idx)

    def _index_counter_keys(self, counter, special_tokens, max_size, min_freq):
        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)
        for token, freq in token_freqs:
            if freq < min_freq or len(self.idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self.idx_to_token[len(self._idx_to_token)] = token
                self._token_to_idx[token] = len(self._token_to_idx)

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError(
                'User-specified token_to_idx mapping can only contain '
                'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError(
                'User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
                self.token_to_idx):
            raise ValueError(
                'User-specified indices must not be < 0 or >= the number of tokens '
                'that will be in the vocabulary. The current vocab contains {}'
                'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    def to_tokens(self, indices):
        """
        Maps the input indices to token list.

        Args:
            indices (int|list[int]|tuple[int]|numpy.ndarray): The input indice(s) for mapping.
                Must be an `int` or 1D `list[int]`|`tuple[int]`|`numpy.ndarray`.

        Returns:
            str|list[str]: Obtained token(s). If `indices` is an integer, it
            will return a str. If `indices` is a list/tuple of integers, it will
            return a list of str.
        """
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self._idx_to_token[int(index)] for index in indices]
        return self._idx_to_token[indices]

    def to_ids(self, tokens):
        """
        Maps the input tokens into indices.

        Args:
            tokens (str|list[str]|tuple[str], optional): The input token(s) for
                mapping.

        Returns:
            int|list[int]: Obationed indice(s). If `tokens` is a str, it will
            return an integer. If `tokens` is a list/tuple of str, it will
            return a list of integers.
        """
        if isinstance(tokens, (list, tuple)):
            return [self.to_ids(token) for token in tokens]
        return self._token_to_idx.get(tokens, self.unk_token)

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self.unk_token)
        return [self.__getitem__(token) for token in tokens]

    def __contains__(self, token):
        return token in self._token_to_idx

    @property
    def idx_to_token(self):
        # Returns index-token dict
        return self._idx_to_token

    @property
    def token_to_idx(self):
        # Return token-index dict
        return self._token_to_idx

    def to_json(self, path=None):
        """
        Summarizes some information of vocab as JSON string. If path is gaven,
        the JSON string will be saved into files. The JSON string and the saved
        file all can be used to reconstruct the :class:`Vocab` by calling
        :meth:`from_json` method.

        Args:
            path (str, optional): The path to save JSON string. If None, the
                JSON will not be saved. Default: None.

        Returns:
            str: The JSON string including information of vocab.
        """
        vocab_dict = {}
        vocab_dict['idx_to_token'] = dict(self._idx_to_token)
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['special_token'] = self.special_token_dict
        json_str = json.dumps(vocab_dict)
        if path:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str):
        """
        Loads :class:`Vocab` from JSON string or JSON file, which is gotten by
        calling :meth:`to_json` method.

        Args:
            json_str (str): JSON string or file path of JSON string.

        Returns:
            Vocab: An instance of :class:`Vocab` generated from information
            contained in JSON string.

        """
        if os.path.isfile(json_str):
            with io.open(json_str, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        special_tokens = vocab_dict.get('special_token', dict())
        vocab = cls(counter=None, token_to_idx=token_to_idx, **special_tokens)

        return vocab

    @classmethod
    def from_dict(cls,
                  token_to_idx,
                  unk_token=None,
                  pad_token=None,
                  bos_token=None,
                  eos_token=None,
                  **kwargs):
        """
        Builds the :class:`Vocab` from a dict.

        Args:
            token_to_idx (dict): A dict describes the mapping relationship between
                tokens and indices.
            unk_token (str, optional): The special token for unknow token. If
                no need, it also could be None. Default: None.
            pad_token (str, optional): The special token for padding token. If
                no need, it also could be None. Default: None.
            bos_token (str, optional): The special token for bos token. If no
                need, it also could be None. Default: None.
            eos_token (str, optional): The special token for eos token. If no
                need, it also could be None. Default: None.

            kwargs (dict): Keyword arguments ending with `_token`. It can be
                used to specify further special tokens that will be exposed as
                attribute of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of :class:`Vocab` generated from the given dict
            and special tokens.
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
    def build_vocab(iterator,
                    max_size=None,
                    min_freq=1,
                    token_to_idx=None,
                    unk_token=None,
                    pad_token=None,
                    bos_token=None,
                    eos_token=None,
                    **kwargs):
        """
        Builds the :class:`Vocab` accoring to given iterator and other
        information. Firstly, iterate over the `iterator` to construct a
        :class:`collections.Counter` and used to init the as  :class:`Vocab`.

        Args:
            iterator (collections.Iterable): Iterator of tokens. Each element
                should be a list of tokens if wordlevel vocab is needed.
            max_size (int, optional): The max size of vocab, not including
                special tokens. Default: None.
            min_freq (int, optional): Ignore tokens whose frequencies are less
                than `min_freq`. Default: 1.
            token_to_idx (dict, optional): A dict specifies the mapping
                relationship between tokens and indices to be used. If provided,
                adjust the tokens and indices mapping according to it. If None,
                counter must be provided. Default: None.
            unk_token (str, optional): The special token for unknow token
                '<unk>'. If no need, it also could be None. Default: None.
            pad_token (str, optional): The special token for padding token
                '<pad>'. If no need, it also could be None. Default: None.
            bos_token (str, optional): The special token for bos token '<bos>'.
                If no need, it also could be None. Default: None.
            eos_token (str, optional): The special token for eos token '<eos>'.
                If no need, it also could be None. Default: None.

            kwargs (dict): Keyword arguments ending with `_token`. It can be
                used to specify further special tokens that will be exposed as
                attribute of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of :class:`Vocab` generated from given iterator
            and other informations.
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

    def save_vocabulary(self, path):
        """Save the vocabulary to a file. """
        with open(path, 'w') as writer:
            for idx in range(len(self._idx_to_token)):
                writer.write(self._idx_to_token[idx] + '\n')

    def load_vocabulary(vocab_file,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Builds the :class:`Vocab` from a file reserving all tokens by calling
        :meth:`Vocab.from_dict` method. The file contains a token per line, and
        the line index would be the index of corresponding token.

        Args:
            filepath (str): the path of file to construct vocabulary.
            unk_token (str, optional): special token for unknown token. If no
                need, it also could be None. Default: None.
            pad_token (str, optional): special token for padding token. If no
                need, it also could be None. Default: None.
            bos_token (str, optional): special token for bos token. If no need,
                it also could be None. Default: None.
            eos_token (str, optional): special token for eos token. If no need,
                it also could be None. Default: None.

            kwargs (dict): Keyword arguments ending with `_token`. It can be
                used to specify further special tokens that will be exposed as
                attribute of the vocabulary and associated with an index.

        Returns:
            Vocab: An instance of :class:`Vocab` generated from the given file.
        """

        token_lst = []
        token_to_idx = collections.OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            index = 0
            for line in f.readlines():
                token = line.rstrip('\n').split('\t')[0]
                token_to_idx[token] = int(index)
                index += 1
        token_lst = list(token_to_idx.keys())
        token_to_idx = {token: idx for idx, token in enumerate(token_lst)}
        vocab = Vocab.from_dict(token_to_idx,
                                unk_token=unk_token,
                                pad_token=pad_token,
                                bos_token=bos_token,
                                eos_token=eos_token,
                                **kwargs)
        return vocab

    def get_unk_token_id(self):
        return self.token_to_idx[
            self.unk_token] if self.unk_token is not None else self.unk_token

    def get_bos_token_id(self):
        return self.token_to_idx[
            self.bos_token] if self.bos_token is not None else self.bos_token

    def get_eos_token_id(self):
        return self.token_to_idx[
            self.eos_token] if self.eos_token is not None else self.eos_token

    def get_pad_token_id(self):
        return self.token_to_idx[
            self.pad_token] if self.pad_token is not None else self.pad_token
