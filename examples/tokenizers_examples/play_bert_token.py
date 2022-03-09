'''
Author: jianzhnie
Date: 2022-03-09 10:58:00
LastEditTime: 2022-03-09 11:05:07
LastEditors: jianzhnie
Description:

'''
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

if __name__ == '__main__':
    bert_tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
    bert_tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()])
    bert_tokenizer.pre_tokenizer = Whitespace()

    bert_tokenizer.post_processor = TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[
            ('[CLS]', 1),
            ('[SEP]', 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size=30522,
        special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    files = [
        f'/home/robin/jianzh/nlp-toolkit/data/ptb/ptb.{split}.txt'
        for split in ['test', 'train', 'valid']
    ]
    bert_tokenizer.train(files, trainer)

    bert_tokenizer.save(
        '/home/robin/jianzh/nlp-toolkit/data/ptb/bert-token.json')
