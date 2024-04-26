"""在自然语言处理（NLP）和文本处理领域，有许多不同的词元化方法，常用的包括以下几种：

1.空格分词（Whitespace Tokenization）：

    将文本按照空格进行分割，将每个分割后的词作为一个词元。例如，"Hello world!"会被分割为[“Hello”, “world!”]。

2.分词器（Tokenizer）：

    使用专门的分词器工具，如NLTK（Natural Language Toolkit）或spaCy，在文本中识别和分割单词。这些分词器可以处理更复杂的分割规则，例如处理标点符号、缩写词等。

3.n-gram分词：

    将文本切分为连续的n个词的组合，这些组合称为n-gram。常见的是二元（bigram）和三元（trigram）分词。例如，"Hello world!"的bigram分词为[“Hello world”, “world!”]。

4.字符级别分词（Character-level Tokenization）：

    将文本按照字符进行分割，将每个字符作为一个词元。这种方法对于处理字符级别的任务（如拼写检查、机器翻译等）很有用。

5.子词（Subword）分词：

    将单词切分为更小的单元，如词根、前缀或后缀等。这种方法可以处理未登录词（out-of-vocabulary）问题，并且对于具有复杂形态的语言（如德语、芬兰语）也很有效。常见的子词分词算法有Byte-Pair Encoding（BPE）和SentencePiece。

6.正则表达式分词（Regular Expression Tokenization）：

    使用正则表达式来定义词元的分割规则。这样可以灵活地根据需要定义分割规则，适用于特定领域的文本处理任务。

7.词干提取（Stemming）和词形还原（Lemmatization）：

    词干提取和词形还原是将单词转换为其基本形式的方法，通常用于减少词汇的复杂性。它们可以与其他词元化方法结合使用。

8.专业领域的自定义分词：

    对于特定领域的文本，有时需要开发自定义的分词方法，以处理领域特有的术语和规则。
这些词元化方法可以根据任务和语言的不同进行组合和调整，选择合适的方法有助于提高NLP应用的性能和效果。
"""

import re
from typing import List, Union

import jieba

from nlptoolkit.data.vocab import Vocab


class BaseTokenizer(object):

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class Tokenizer:
    """Tokenizes text lines into word or character tokens.

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
        """Tokenize the input sentence into word or character tokens.

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
        # 将句子中的特殊字符（如星号、引号、换行符、反斜杠、加号、减号、斜杠、等号、括号、单引号、冒号、方括号、竖线、感叹号和分号）
        # 替换为一个空格。

        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", ' ',
                          str(sentence))
        # 将连续多个空格替换为一个空格。这有助于将多个连续空格合并成一个。
        sentence = re.sub(r'[ ]+', ' ', sentence)
        # 将连续多个感叹号替换为一个感叹号，类似地，后面的行也分别用于处理连续的逗号和问号。
        sentence = re.sub(r'\!+', '!', sentence)
        sentence = re.sub(r'\,+', ',', sentence)
        sentence = re.sub(r'\?+', '?', sentence)
        # 替换非字母字符（包括数字、符号和空格）为一个空格。这将确保句子中只包含字母字符。
        sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
        # 将整个句子转换为小写字母，以确保文本的一致性，因为在自然语言处理任务中通常不区分大小写。
        sentence = sentence.lower()
        # 接下来，根据指定的token类型，函数将句子分割成单词或字符，并返回结果：
        if token == 'word':
            return sentence.split()
        elif token == 'char':
            return [list(word) for word in sentence.split()]
        else:
            raise ValueError('Unknown token type: ' + token)


class JiebaTokenizer(BaseTokenizer):
    """Constructs a tokenizer based on `jieba.

    <https://github.com/fxsjy/jieba>`__. It supports :meth:`cut` method to
    split the text to tokens, and :meth:`encode` method to convert text to token
    ids.

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
        """The method used to cut the text to tokens.

        Args:
            sentence(str): The text that needs to be cut.
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
            sentence(str): The text that needs to be cut.
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
