'''
Author: jianzhnie
Date: 2021-12-22 16:34:43
LastEditTime: 2021-12-24 18:32:39
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn as nn
from nlptoolkit.models.transformer.common import EncoderBlock


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_hiddens, n_segments, max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(n_segments, num_hiddens)
        self.pos_embedding = nn.Embedding(max_len, num_hiddens)

        self.norm = nn.LayerNorm(num_hiddens)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(
            x)  # (seq_len, ) --> (batch_size, seq_len)
        embedding = self.token_embedding(x) + self.pos_embedding(
            pos) + self.segment_embedding(seg)
        return self.norm(embedding)


class BERTEncoder(nn.Module):
    """BERT encoder.

    - 与原始 TransformerEncoder不同，BERTEncoder使用片段嵌入和可学习的位置嵌入。

    Defined in :numref:`subsec_bert_input_rep`
    """
    def __init__(self,
                 vocab_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 num_layers,
                 dropout,
                 max_len=1000,
                 key_size=768,
                 query_size=768,
                 value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'{i}',
                EncoderBlock(key_size=key_size,
                             query_size=query_size,
                             value_size=value_size,
                             num_hiddens=num_hiddens,
                             norm_shape=norm_shape,
                             ffn_num_input=ffn_num_input,
                             ffn_num_hiddens=ffn_num_hiddens,
                             num_heads=num_heads,
                             dropout=dropout,
                             use_bias=True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """The masked language model task of BERT.

    - 80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
    - 10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”）；
    - 10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）

    实现了MaskLM类来预测BERT预训练的掩蔽语言模型任务中的掩蔽标记。
    - 预测使用单隐藏层的多层感知机（self.mlp）。在前向推断中，它需要两个输入：
        - BERTEncoder的编码结果和用于预测的词元位置。
        - 输出是这些位置的预测结果。


    Defined in :numref:`subsec_bert_input_rep`
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT.

    - 为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类任务——下一句预测。
    - 在为预训练生成句子对时，有一半的时间它们确实是标签为“真”的连续句子；
    - 在另一半的时间里，第二个句子是从语料库中随机抽取的，标记为“假”。

    - NextSentencePred类使用单隐藏层的多层感知机来预测第二个句子是否是BERT输入序列中第一个句子的下一个句子。
    - 由于Transformer编码器中的自注意力，特殊词元“<cls>”的BERT表示已经对输入的两个句子进行了编码。过程如下:
        - step1: 带 “<cls>”标记的词元 X ;
        - step2: encoded_X = BertEncoder(X) 编码后的词元
        - step3: output = MLP(encoded_X[:, 0, :])  BertModel 的 Head, 0 是“<cls>”标记的索引
        - step4: output = NextSentencePred(output)  单隐藏层的 MLP 预测下一个句子.

    Defined in :numref:`subsec_mlm`
    """
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)


class BERTModel(nn.Module):
    """The BERT model.

    - 定义BERTModel类, 实例化三个类:
        - BERTEncoder
        - MaskLM
        - NextSentencePred
    - 前向推断返回编码后的BERT表示encoded_X、掩蔽语言模型预测mlm_Y_hat和下一句预测nsp_Y_hat。

    Defined in :numref:`subsec_nsp`
    """
    def __init__(self,
                 vocab_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 num_layers,
                 dropout,
                 max_len=1000,
                 key_size=768,
                 query_size=768,
                 value_size=768,
                 hid_in_features=768,
                 mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size,
                                   num_hiddens,
                                   norm_shape,
                                   ffn_num_input,
                                   ffn_num_hiddens,
                                   num_heads,
                                   num_layers,
                                   dropout,
                                   max_len=max_len,
                                   key_size=key_size,
                                   query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__ == '__main__':
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    print(encoder)

    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)

    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)
