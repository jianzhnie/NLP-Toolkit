'''
Author: jianzhnie
Date: 2022-03-04 17:13:34
LastEditTime: 2022-03-04 17:16:27
LastEditors: jianzhnie
Description:

'''
import os
import random

from torch.utils.data import Dataset

from nlptoolkit.data.utils.bert_processor import (get_mlm_data_from_tokens,
                                                  get_nsp_data_from_paragraph,
                                                  pad_bert_inputs)
from nlptoolkit.data.vocab import Vocab, tokenize


class BertDataSet(Dataset):
    def __init__(self,
                 data_dir,
                 max_len,
                 reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'],
                 encoding='utf-8') -> None:
        super().__init__()
        self.max_len = max_len
        self.data_dir = data_dir
        self.encoding = encoding

        paragraphs = self.read_wiki(data_dir)
        paragraphs = [
            tokenize(paragraph, token='word') for paragraph in paragraphs
        ]
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph
        ]

        self.vocab = Vocab(sentences,
                           min_freq=5,
                           reserved_tokens=reserved_tokens)

        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(
                get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))
        # examples: ([token_a, token_b], [0,0,1, ], True)
        # 获取遮蔽语言模型任务的数据
        examples = [(get_mlm_data_from_tokens(tokens, self.vocab) +
                     (segments, is_next))
                    for tokens, segments, is_next in examples]
        # [vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels] + (segments, is_next)]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

    def read_wiki(self, data_dir, encoding):
        file_name = os.path.join(data_dir, 'wiki.train.tokens')
        with open(file_name, 'r', encoding=encoding) as f:
            lines = f.readlines()
        # 大写字母转换为小写字母
        paragraphs = [
            line.strip().lower().split(' . ') for line in lines
            if len(line.split(' . ')) >= 2
        ]
        random.shuffle(paragraphs)
        return paragraphs
