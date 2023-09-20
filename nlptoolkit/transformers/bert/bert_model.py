'''
Author: jianzhnie
Date: 2021-12-22 16:34:43
LastEditTime: 2021-12-24 18:32:39
LastEditors: jianzhnie
Description:

'''

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from nlptoolkit.transformers.vanilla import (TransformerEncoder,
                                             TransformerEncoderLayer)


class BertEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 type_vocab_size,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=1000,
                 position_embedding_type='absolute'):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           hidden_size,
                                           padding_idx=0)
        self.position_embedding = nn.Embedding(max_position_embeddings,
                                               hidden_size)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.position_embedding_type = position_embedding_type
        self.register_buffer('position_ids',
                             torch.arange(max_position_embeddings).expand(
                                 (1, -1)),
                             persistent=False)
        self.register_buffer('token_type_ids',
                             torch.zeros(self.position_ids.size(),
                                         dtype=torch.long),
                             persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape,
                                             dtype=torch.long,
                                             device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embedding(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embedding(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MaskLM(nn.Module):
    """The masked language model task of BERT.

    - 80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
    - 10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”）；
    - 10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）

    实现了MaskLM类来预测BERT预训练的掩蔽语言模型任务中的掩蔽标记。
    - 预测使用单隐藏层的多层感知机（self.mlp）。在前向推断中，它需要两个输入：
        - BERTEncoder的编码结果和用于预测的词元位置。
        - 输出是这些位置的预测结果。

    """
    def __init__(self, vocab_size, hidden_size, d_model=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden_size), nn.ReLU(),
                                 nn.LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, vocab_size))

    def forward(self, inputs, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = inputs.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = inputs[batch_idx, pred_positions]
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
        - step1: 输入带 “<cls>”标记的词元 inputs ;
        - step2: encoded_feature = BertEncoder(inputs) 编码后的词元
        - step3: output = MLP(encoded_feature[:, 0, :])  BertModel 的 Head, 0 是“<cls>”标记的索引
        - step4: output = NextSentencePred(output)  单隐藏层的 MLP 预测下一个句子.

    """
    def __init__(self, d_model, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(d_model, 2)

    def forward(self, inputs):
        # X shape: (batch size, `hidden_size`)
        return self.output(inputs)


class BERTModel(nn.Module):
    """BERT encoder.

    - 与原始 TransformerEncoder不同，BERTEncoder使用片段嵌入和可学习的位置嵌入。
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_segments: int,
        max_len: int = 1000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        mlm_in_features: int = 512,
        nsp_in_features: int = 512,
    ):
        super(BERTModel, self).__init__()
        self.embed_layer = BertEmbedding(vocab_size, hidden_size, n_segments,
                                         max_len)
        encoder_layer = TransformerEncoderLayer(d_model, num_heads,
                                                dim_feedforward, dropout,
                                                activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers,
                                          encoder_norm)
        self.mlm = MaskLM(vocab_size, hidden_size, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(
        self,
        tokens: Tensor,
        segments: Tensor,
        pred_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform a forward pass through the Transformer model.

        Args:
            src (Tensor): The sequence to the encoder (required).
            src_mask (Optional[Tensor]): The additive mask for the src sequence (optional).
            src_key_padding_mask (Optional[Tensor]): The ByteTensor mask for src keys per batch (optional).

        Returns:
            Tensor: The output tensor of shape (T, N, E).

        Shape:
            - src: (S, N, E).
            - src_mask: (S, S).
            - src_key_padding_mask: (N, S).
        """
        embeddings = self.embed_layer(tokens, segments)
        output = self.encoder(embeddings)
        if pred_positions is not None:
            mlm_pred = self.mlm(output, pred_positions)
        else:
            mlm_pred = None
        nsp_pred = self.nsp(output[:, 0, :])
        return output, mlm_pred, nsp_pred

    def _reset_parameters(self):
        """
        Initialize parameters in the transformer model using Xavier initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


if __name__ == '__main__':
    vocab_size, hidden_size, ffn_hidden_size, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTModel(vocab_size, hidden_size, norm_shape, ffn_num_input,
                        ffn_hidden_size, num_heads, num_layers, dropout)
    print(encoder)

    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

    mlm = MaskLM(vocab_size, hidden_size)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)

    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)
