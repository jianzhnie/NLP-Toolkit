import collections
import logging
import os
import unicodedata
from io import open
from typing import List, Optional, OrderedDict, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file': {
        'bert-base-uncased':
        'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt',
        'bert-large-uncased':
        'https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt',
        'bert-base-cased':
        'https://huggingface.co/bert-base-cased/resolve/main/vocab.txt',
        'bert-large-cased':
        'https://huggingface.co/bert-large-cased/resolve/main/vocab.txt',
        'bert-base-multilingual-uncased':
        ('https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt'
         ),
        'bert-base-multilingual-cased':
        'https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt',
        'bert-base-chinese':
        'https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt',
        'bert-base-german-cased':
        'https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt',
        'bert-large-uncased-whole-word-masking':
        ('https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt'
         ),
        'bert-large-cased-whole-word-masking':
        ('https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt'
         ),
        'bert-large-uncased-whole-word-masking-finetuned-squad':
        ('https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt'
         ),
        'bert-large-cased-whole-word-masking-finetuned-squad':
        ('https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt'
         ),
        'bert-base-cased-finetuned-mrpc':
        ('https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt'
         ),
        'bert-base-german-dbmdz-cased':
        'https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt',
        'bert-base-german-dbmdz-uncased':
        ('https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt'
         ),
        'TurkuNLP/bert-base-finnish-cased-v1':
        ('https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt'
         ),
        'TurkuNLP/bert-base-finnish-uncased-v1':
        ('https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt'
         ),
        'wietsedv/bert-base-dutch-cased':
        ('https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt'
         ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
    'bert-base-german-cased': 512,
    'bert-large-uncased-whole-word-masking': 512,
    'bert-large-cased-whole-word-masking': 512,
    'bert-large-uncased-whole-word-masking-finetuned-squad': 512,
    'bert-large-cased-whole-word-masking-finetuned-squad': 512,
    'bert-base-cased-finetuned-mrpc': 512,
    'bert-base-german-dbmdz-cased': 512,
    'bert-base-german-dbmdz-uncased': 512,
    'TurkuNLP/bert-base-finnish-cased-v1': 512,
    'TurkuNLP/bert-base-finnish-uncased-v1': 512,
    'wietsedv/bert-base-dutch-cased': 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    'bert-base-uncased': {
        'do_lower_case': True
    },
    'bert-large-uncased': {
        'do_lower_case': True
    },
    'bert-base-cased': {
        'do_lower_case': False
    },
    'bert-large-cased': {
        'do_lower_case': False
    },
    'bert-base-multilingual-uncased': {
        'do_lower_case': True
    },
    'bert-base-multilingual-cased': {
        'do_lower_case': False
    },
    'bert-base-chinese': {
        'do_lower_case': False
    },
    'bert-base-german-cased': {
        'do_lower_case': False
    },
    'bert-large-uncased-whole-word-masking': {
        'do_lower_case': True
    },
    'bert-large-cased-whole-word-masking': {
        'do_lower_case': False
    },
    'bert-large-uncased-whole-word-masking-finetuned-squad': {
        'do_lower_case': True
    },
    'bert-large-cased-whole-word-masking-finetuned-squad': {
        'do_lower_case': False
    },
    'bert-base-cased-finetuned-mrpc': {
        'do_lower_case': False
    },
    'bert-base-german-dbmdz-cased': {
        'do_lower_case': False
    },
    'bert-base-german-dbmdz-uncased': {
        'do_lower_case': True
    },
    'TurkuNLP/bert-base-finnish-cased-v1': {
        'do_lower_case': False
    },
    'TurkuNLP/bert-base-finnish-uncased-v1': {
        'do_lower_case': True
    },
    'wietsedv/bert-base-dutch-cased': {
        'do_lower_case': False
    },
}


def convert_to_unicode(text: Union[str, bytes]) -> str:
    """
    Converts the input `text` to Unicode assuming UTF-8 encoding.

    Args:
        text (Union[str, bytes]): The input text, which can be either a string or bytes.

    Returns:
        str: Unicode representation of the input text.

    Raises:
        ValueError: If the input `text` is not a supported string type.
    """
    if isinstance(text, str):
        # If the input is already a Unicode string, return it as it is.
        return text
    elif isinstance(text, bytes):
        # If the input is a byte string, decode it using UTF-8 encoding.
        return text.decode('utf-8', 'ignore')
    else:
        # If the input is neither a string nor bytes, raise a ValueError.
        raise ValueError('Unsupported string type: %s' % (type(text)))


def _is_whitespace(char: str) -> bool:
    """
    Checks whether the input character is a whitespace character.

    Args:
        char (str): Input character to be checked.

    Returns:
        bool: True if the character is a whitespace character, False otherwise.
    """
    if char in [' ', '\t', '\n', '\r']:
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char: str) -> bool:
    """
    Checks whether the input character is a control character.

    Args:
        char (str): Input character to be checked.

    Returns:
        bool: True if the character is a control character, False otherwise.
    """
    if char in ['\t', '\n', '\r']:
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char: str) -> bool:
    """
    Checks whether the input character is a punctuation character.

    Args:
        char (str): Input character to be checked.

    Returns:
        bool: True if the character is a punctuation character, False otherwise.
    """
    cp = ord(char)
    # ASCII punctuation characters (33-47, 58-64, 91-96, 123-126)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def load_vocab(vocab_file: str) -> OrderedDict[str, int]:
    """
    Loads a vocabulary file into an ordered dictionary where keys are tokens and values are indices.

    Args:
        vocab_file (str): Path to the vocabulary file.

    Returns:
        OrderedDict[str, int]: An ordered dictionary mapping tokens to their indices.
    """
    vocab = collections.OrderedDict()
    try:
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            tokens = reader.readlines()
            for index, token in enumerate(tokens):
                # Strip whitespace and add token to the vocabulary with its index.
                token = token.rstrip('\n')
                vocab[token] = index
    except FileNotFoundError:
        # Handle file not found error and provide a meaningful message.
        raise FileNotFoundError(f'Vocabulary file not found: {vocab_file}')
    except Exception as e:
        # Handle other exceptions and provide an error message.
        raise Exception(f'Error loading vocabulary file: {str(e)}')

    return vocab


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenizes a piece of text based on whitespace.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        List[str]: A list of tokens obtained by splitting the input text on whitespace.
    """
    # Remove leading and trailing whitespaces and split the text into tokens using whitespace as delimiter.
    tokens = text.strip().split()
    return tokens


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """
    def __init__(
        self,
        do_lower_case: bool = True,
        never_split: List[str] = [
            '[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'
        ],
        tokenize_chinese_chars: bool = True,
        strip_accents: bool = None,
        do_split_on_punc: bool = True,
    ):
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text: str, never_split=None) -> List[str]:
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            text (str): Input text to be tokenized.
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.

        Returns:
            List[str]: A list of tokens obtained after tokenization.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(
            set(never_split)) if never_split else self.never_split
        # Clean the text by removing invalid characters and performing whitespace cleanup.
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).

        # Tokenize Chinese characters by adding whitespace around them.
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        # prevents treating the same character with different unicode codepoints as different characters
        # Tokenize the text based on whitespace and punctuation.
        unicode_normalized_text = unicodedata.normalize('NFC', text)
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                # Lowercase the token if do_lower_case is True and it's not a special token.
                if self.do_lower_case:
                    token = token.lower()
                    # Remove accents from the token.
                    if self.strip_accents:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token, never_split)

            # Split the token based on punctuation.
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # Perform whitespace tokenization again and return the final list of tokens.
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str) -> str:
        """
        Strips accents from a piece of text.

        Args:
            text (str): Input text from which accents need to be stripped.

        Returns:
            str: Text with accents removed.
        """
        # Normalize the text to decompose characters into base characters and combining characters.
        normalized_text = unicodedata.normalize('NFD', text)
        output = []
        for char in normalized_text:
            # Get the Unicode category of the character.
            cat = unicodedata.category(char)
            # If the character is a Nonspacing_Mark, skip it (remove the accent).
            if cat != 'Mn':
                output.append(char)
        # Join the characters back into a string and return.
        return ''.join(output)

    def _run_split_on_punc(self, text: str, never_split=None) -> List[str]:
        """
        Splits punctuation on a piece of text.

        Args:
            text (str): Input text to be split.

        Returns:
            list: A list of tokens obtained by splitting the input text on punctuation.
        """
        # If the text is a special token, return it as a single-token list.
        if not self.do_split_on_punc or (never_split is not None
                                         and text in never_split):
            return [text]

        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                # If the character is punctuation, start a new token.
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    # If a new token is starting, append an empty list to output.
                    output.append([])
                start_new_word = False
                # Add the character to the last token in output.
                output[-1].append(char)
            i += 1

        # Join the characters in each token and return the list of tokens.
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        """
        Adds whitespace around any CJK character.

        Args:
            text (str): Input text in which whitespace needs to be added around CJK characters.

        Returns:
            str: Text with whitespace added around CJK characters.
        """
        output = []
        for char in text:
            # Get the Unicode code point of the character.
            cp = ord(char)
            if self._is_chinese_char(cp):
                # If the character is a CJK character, add whitespace around it.
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                # If the character is not a CJK character, keep it as it is.
                output.append(char)
        # Join the characters back into a string and return.
        return ''.join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        """
        Checks whether CP is the codepoint of a CJK character.

        Args:
            cp (int): Unicode code point of a character.

        Returns:
            bool: True if the code point represents a CJK character, False otherwise.
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        # CJK Unicode blocks where Chinese, Japanese, and Korean characters are encoded.
        cjk_ranges = [
            (0x4E00, 0x9FFF),  # Common CJK Ideographs
            (0x3400, 0x4DBF),  # Extension A
            (0x20000, 0x2A6DF),  # Extension B
            (0x2A700, 0x2B73F),  # Extension C
            (0x2B740, 0x2B81F),  # Extension D
            (0x2B820, 0x2CEAF),  # Extension E
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
            (0x2F800, 0x2FA1F)  # CJK Compatibility Ideographs Supplement
        ]

        # Check if the code point falls within any of the specified CJK ranges.
        for start, end in cjk_ranges:
            if start <= cp <= end:
                return True

        # If the code point is not within any CJK range, return False.
        return False

    def _clean_text(self, text: str) -> str:
        """
        Performs invalid character removal and whitespace cleanup on text.

        Args:
            text (str): Input text to be cleaned.

        Returns:
            str: Cleaned text after removing invalid characters and performing whitespace cleanup.
        """
        output = []
        for char in text:
            cp = ord(char)
            # Skip invalid characters and control characters.
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            # Replace whitespace characters with a single space.
            if _is_whitespace(char):
                output.append(' ')
            else:
                # Keep valid characters as they are.
                output.append(char)
        # Join the characters back into a string and return.
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        """
        Initializes the WordPiece tokenizer.

        Args:
            vocab (dict): Vocabulary mapping from subwords to indices.
            unk_token (str): Token representing unknown words not in the vocabulary.
            max_input_chars_per_word (int): Maximum number of characters considered as a single word.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> list:
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # If the token is too long, replace it with the unknown token.
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    # If no valid subword is found, replace it with the unknown token.
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]',
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. "
                'To load the vocabulary from a Google pretrained model, use '
                '`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      unk_token=str(unk_token))

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        """
        Tokenizes a piece of text into its WordPiece tokens.

        Args:
            text (str): Input text to be tokenized.

        Returns:
            list: A list of WordPiece tokens.
        """
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                    text,
                    never_split=self.all_special_tokens
                    if not split_special_tokens else None):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 +
                                                        sep) * [1]

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + '-' if filename_prefix else '') +
                VOCAB_FILES_NAMES['vocab_file'])
        else:
            vocab_file = (filename_prefix +
                          '-' if filename_prefix else '') + save_directory
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.vocab.items(),
                                             key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f'Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive.'
                        ' Please check that the vocabulary is not corrupted!')
                    index = token_index
                writer.write(token + '\n')
                index += 1
        return (vocab_file, )
