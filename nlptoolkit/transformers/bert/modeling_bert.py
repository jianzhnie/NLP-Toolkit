import torch
import torch.nn as nn
from transformers import PretrainedModel


class BertEmbeddings(nn.Module):
    """Include embeddings from word, position and token_type embeddings."""
    def __init__(self,
                 vovab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(vovab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype='int64')
            seq_length = torch.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype='int64')

        input_embedings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    """Pool the result of BertEncoder."""
    def __init__(self, hidden_size, pool_act='tanh'):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPretrainedModel(PretrainedModel):
    """An abstract class for pretrained BERT models.

    It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = 'model_config.json'
    pretrained_init_configuration = {
        'bert-base-uncased': {
            'vocab_size': 30522,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-large-uncased': {
            'vocab_size': 30522,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-base-multilingual-uncased': {
            'vocab_size': 105879,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-base-cased': {
            'vocab_size': 28996,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-base-chinese': {
            'vocab_size': 21128,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-base-multilingual-cased': {
            'vocab_size': 119547,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-large-cased': {
            'vocab_size': 28996,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-wwm-chinese': {
            'vocab_size': 21128,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'bert-wwm-ext-chinese': {
            'vocab_size': 21128,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'macbert-base-chinese': {
            'vocab_size': 21128,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'macbert-large-chinese': {
            'vocab_size': 21128,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
        'simbert-base-chinese': {
            'vocab_size': 13685,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'pad_token_id': 0,
        },
    }
    resource_files_names = {'model_state': 'model_state.pdparams'}

    base_model_prefix = 'bert'

    def init_weights(self, layer):
        """Initialization hook."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, torch.Tensor):
                layer.weight.set_value(
                    torch.tensor.normal(mean=0.0,
                                        std=self.initializer_range if hasattr(
                                            self, 'initializer_range') else
                                        self.bert.config['initializer_range'],
                                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class BertModel(BertPretrainedModel):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 pool_act='tanh') -> None:

        super(BertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, dropout_prob,
                                         max_position_embeddings,
                                         type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_prob,
            activation=hidden_act,
            norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False):

        if attention_mask is None:
            attention_mask = torch.cast(
                input_ids == self.pad_token_id,
                dtype=torch.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2])

        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids)

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])

        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class BertForQuestionAnswering(BertPretrainedModel):
    """Bert Model with a linear layer on top of the hidden-states output to
    compute `span_start_logits` and `span_end_logits`, designed for question-
    answering tasks like SQuAD.

    Args:
        bert (:class:`BertModel`):
            An instance of BertModel.
        dropout (float, optional):
            The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Defaults to `None`.
    """
    def __init__(self, bert, dropout=None):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  bert.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.bert.config['hidden_size'], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The BertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
        """

        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = torch.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = torch.unstack(x=logits, axis=0)

        return start_logits, end_logits


class BertForSequenceClassification(BertPretrainedModel):
    """Bert Model with a linear layer on top of the output layer, designed for
    sequence classification/regression tasks like GLUE tasks.

    Args:
        bert (:class:`BertModel`):
            An instance of BertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Defaults to None.
    """
    def __init__(self, bert, num_classes=2, dropout=None):
        super(BertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  bert.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.bert.config['hidden_size'],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The BertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (list, optional):
                See :class:`BertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        """

        _, pooled_output = self.bert(input_ids,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertForTokenClassification(BertPretrainedModel):
    """Bert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        bert (:class:`BertModel`):
            An instance of BertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Defaults to None.
    """
    def __init__(self, bert, num_classes=2, dropout=None):
        super(BertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  bert.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.bert.config['hidden_size'],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The BertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (list, optional):
                See :class:`BertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        """
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class BertLMPredictionHead(nn.Module):
    """Bert Model with a `language modeling` head on top for CLM fine-
    tuning."""
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = torch.reshape(hidden_states,
                                          [-1, hidden_states.shape[-1]])
            hidden_states = torch.tensor.gather(hidden_states,
                                                masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, activation, embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            activation=activation,
            embedding_weights=embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class BertForMaskedLM(BertPretrainedModel):
    """Bert Model with a `masked language modeling` head on top.

    Args:
        bert (:class:`BertModel`):
            An instance of :class:`BertModel`.
    """
    def __init__(self, bert):
        super(BertForMaskedLM, self).__init__()
        self.bert = bert
        self.cls = BertOnlyMLMHead(
            self.bert.config['hidden_size'],
            self.bert.config['vocab_size'],
            self.bert.config['hidden_act'],
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids (Tensor, optional):
                See :class:`BertModel`.
            attention_mask (Tensor, optional):
                See :class:`BertModel`.

        Returns:
            Tensor: Returns tensor `prediction_scores`, The scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        """

        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions=None)
        return prediction_scores
