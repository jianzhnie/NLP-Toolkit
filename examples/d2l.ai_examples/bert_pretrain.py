'''
Author: jianzhnie
Date: 2021-12-24 14:24:31
LastEditTime: 2021-12-24 16:31:24
LastEditors: jianzhnie
Description:

'''
import sys

import torch
from d2l import torch as d2l
from torch import nn

from nlptoolkit.data.datasets.snli import SNLIBERTDataset
from nlptoolkit.models.transformer.bert.bert_fintune import (
    BERTClassifier, load_pretrained_model)

sys.path.append('../../')

d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')

if __name__ == '__main__':
    devices = d2l.try_all_gpus()
    bert, vocab = load_pretrained_model('bert.small',
                                        num_hiddens=256,
                                        ffn_num_hiddens=512,
                                        num_heads=4,
                                        num_layers=2,
                                        dropout=0.1,
                                        max_len=512,
                                        devices=devices)

    # 如果出现显存不足错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
    batch_size, max_len, num_workers = 64, 128, d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set,
                                             batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set,
                                            batch_size,
                                            num_workers=num_workers)

    net = BERTClassifier(bert)

    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
