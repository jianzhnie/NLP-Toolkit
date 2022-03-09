'''
Author: jianzhnie
Date: 2022-03-08 18:24:19
LastEditTime: 2022-03-09 10:53:48
LastEditors: jianzhnie
Description:

'''
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

if __name__ == '__main__':
    trainer = BpeTrainer(
        special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()

    files = [
        f'/home/robin/jianzh/nlp-toolkit/data/ptb/ptb.{split}.txt'
        for split in ['test', 'train', 'valid']
    ]
    tokenizer.train(files, trainer)
    tokenizer.save(
        '/home/robin/jianzh/nlp-toolkit/data/ptb/tokenizer-ptb.json')

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)

    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    output = normalizer.normalize_str('Héllò hôw are ü?')
    # "Hello how are u?"
    print(output)

    pre_tokenizer = Whitespace()
    output = pre_tokenizer.pre_tokenize_str(
        "Hello! How are you? I'm fine, thank you.")
    # [("Hello", (0, 5)), ("!", (5, 6)), ("How", (7, 10)), ("are", (11, 14)), ("you", (15, 18)),
    #  ("?", (18, 19)), ("I", (20, 21)), ("'", (21, 22)), ('m', (22, 23)), ("fine", (24, 28)),
    #  (",", (28, 29)), ("thank", (30, 35)), ("you", (36, 39)), (".", (39, 40))]
    print(output)

    from tokenizers import pre_tokenizers
    from tokenizers.pre_tokenizers import Digits

    pre_tokenizer = pre_tokenizers.Sequence(
        [Whitespace(), Digits(individual_digits=True)])
    pre_tokenizer.pre_tokenize_str('Call 911!')

    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)],
    )
