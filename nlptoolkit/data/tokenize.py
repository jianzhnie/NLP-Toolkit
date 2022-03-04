'''
Author: jianzhnie
Date: 2021-12-29 16:07:11
LastEditTime: 2022-01-04 10:37:58
LastEditors: jianzhnie
Description:

'''

import re


class Tokenizer(object):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_utils`
    """
    def __init__(self, lang):
        self.lang = lang

    def tokenize(self, sentence, token='word'):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", ' ',
                          str(sentence))
        sentence = re.sub(r'[ ]+', ' ', sentence)
        sentence = re.sub(r'\!+', '!', sentence)
        sentence = re.sub(r'\,+', ',', sentence)
        sentence = re.sub(r'\?+', '?', sentence)
        sentence = sentence.lower()
        assert token in ('word', 'char'), 'Unknown token type: ' + token
        return [
            line.split() if token == 'word' else list(line)
            for line in sentence
        ]
