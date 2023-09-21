'''
Author: jianzhnie
Date: 2021-12-22 16:34:43
LastEditTime: 2021-12-24 18:32:39
LastEditors: jianzhnie
Description:

'''

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions

from nlptoolkit.transformers.vanilla import (TransformerEncoder,
                                             TransformerEncoderLayer)


class BertEmbedding(nn.Module):
    """
    Include embeddings from word, position and token_type embeddings
    """
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 type_vocab_size,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=1000):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings,
                                               hidden_size)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
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
        past_key_values_length: Optional[int] = None,
    ):
        input_shape = input_ids.size()
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

        inputs_embeddings = self.word_embedding(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    """
    Pool the result of BertEncoder.
    """
    def __init__(self, hidden_size):
        """init the bert pooler with config & args/kwargs

        Args:
            config (BertConfig): BertConfig instance. Defaults to None.
        """
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertOutput(nn.Module):
    def __init__(self,
                 intermediate_size,
                 hidden_size,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


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


class BertForNextSentencePred(nn.Module):
    """The next sentence prediction task of BERT.

    - 为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类任务——下一句预测。
    - 在为预训练生成句子对时，有一半的时间它们确实是标签为“真”的连续句子；
    - 在另一半的时间里，第二个句子是从语料库中随机抽取的，标记为“假”。

    - BertForNextSentencePred 类使用单隐藏层的多层感知机来预测第二个句子是否是BERT输入序列中第一个句子的下一个句子。
    - 由于Transformer编码器中的自注意力，特殊词元“<cls>”的BERT表示已经对输入的两个句子进行了编码。过程如下:
        - step1: 输入带 “<cls>”标记的词元 inputs ;
        - step2: encoded_feature = BertEncoder(inputs) 编码后的词元
        - step3: output = MLP(encoded_feature[:, 0, :])  BertModel 的 Head, 0 是“<cls>”标记的索引
        - step4: output = BertForNextSentencePred(output)  单隐藏层的 MLP 预测下一个句子.

    """
    def __init__(self, d_model, **kwargs):
        super(BertForNextSentencePred, self).__init__(**kwargs)
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
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        mlm_in_features: int = 512,
        nsp_in_features: int = 512,
    ):
        super(BERTModel, self).__init__()
        self.embedding_layer = BertEmbedding(vocab_size, hidden_size,
                                             n_segments, max_len)
        encoder_layer = TransformerEncoderLayer(d_model, num_heads,
                                                intermediate_size, dropout,
                                                activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers,
                                          encoder_norm)
        self.mlm = MaskLM(vocab_size, hidden_size, mlm_in_features)
        self.nsp = BertForNextSentencePred(nsp_in_features)
        self.pooler = BertPooler(d_model)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        r"""
        The BertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~torch.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~torch.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~torch.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import torch
                from torch.transformers import BertModel, BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
                model = BertModel.from_pretrained('bert-wwm-chinese')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:torch.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        past_key_values_length = None
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = torch.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = torch.zeros(
                    [batch_size, 1, 1, past_key_values_length],
                    dtype=attention_mask.dtype)
                attention_mask = torch.concat([past_mask, attention_mask],
                                              axis=-1)
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
                    torch.get_default_dtype())
                attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embedding_layer(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

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
