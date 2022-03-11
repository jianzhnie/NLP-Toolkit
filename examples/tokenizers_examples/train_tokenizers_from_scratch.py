'''
Author: jianzhnie
Date: 2022-03-09 11:43:39
LastEditTime: 2022-03-09 15:29:07
LastEditors: jianzhnie
Description:

'''

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import (BpeTrainer, UnigramTrainer, WordLevelTrainer,
                                 WordPieceTrainer)


def prepare_tokenizer_trainer(alg):
    unk_token = '[UNK]'
    special_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=special_tokens)
    elif alg == 'WordPiece':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=special_tokens)
    elif alg == 'WordLevel':
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=special_tokens)
    elif alg == 'Unigram':
        tokenizer = Tokenizer(Unigram(unk_token=unk_token))
        trainer = UnigramTrainer(special_tokens=special_tokens)

    tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[
            ('[CLS]', 1),
            ('[SEP]', 2),
        ],
    )
    return tokenizer, trainer


def train_tokenizer(root, files, alg='WordPiece'):
    tokenizer, trainer = prepare_tokenizer_trainer(alg)
    tokenizer.train(files, trainer)
    tokenizer.save(f'{root}/{alg}-token.json')
    return tokenizer


if __name__ == '__main__':
    root = '/home/robin/jianzh/nlp-toolkit/data/ptb/'
    files = [f'{root}/ptb.{split}.txt' for split in ['test', 'train', 'valid']]
    print(files)
    tokenizer = train_tokenizer(root, files)
    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)
