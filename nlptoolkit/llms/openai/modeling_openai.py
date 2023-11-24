import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import OpenAIGPTPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers.pytorch_utils import Conv1D

from .config_openai import OpenAIGPTConfig


class Attention(nn.Module):

    def __init__(self, config: OpenAIGPTConfig, scale=False):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f'Attention n_embed shape: {self.n_embd} must be divisible by config.n_head {self.n_head}'
            )
        self.n_positions = config.n_positions
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(self.n_positions, self.n_positions)).view(
                1, 1, self.n_positions, self.n_positions),
            persistent=False,
        )
        self.scale = scale
        self.c_attn = Conv1D(self.n_embd * 3, self.n_embd)
        self.c_proj = Conv1D(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        attention_weight = torch.matmul(query, key)
        if self.scale:
            attention_weight = attention_weight / math.sqrt(value.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        casual_mask = self.bias[:, :, :attention_weight.
                                size(-2), :attention_weight.size(-1)]
        attention_weight = attention_weight * casual_mask + -1e4 * (
            1 - casual_mask)

        if attention_mask is not None:
            # Apply the attention mask
            attention_weight = attention_weight + attention_mask

        attention_weight = nn.functional.softmax(attention_weight, dim=-1)
        attention_weight = self.attn_dropout(attention_weight)

        hiddent_states = torch.matmul(attention_weight, value)
        return hiddent_states

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (tensor.size(-2) * tensor.size(-1), )

        return tensor.view(*new_shape)

    def split_heads(self, tensor: torch.Tensor, k=False) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.n_head,
                                          tensor.size(-1) // self.n_head)
        tensor = tensor.view(
            *new_shape)  # in Tensorflow implementation: fct split_states
        if k:
            return tensor.permute(0, 2, 3, 1)
        else:
            return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.c_attn(hidden_state)
        query, key, value = x.split(self.n_embd, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        output = self._attn(query, key, value, attention_mask)

        a = self.merge_heads(output)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):

    def __init__(self, config: OpenAIGPTConfig) -> None:
        super().__init__()
        self.c_fc = Conv1D(4 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)
        self.act = ACT2FN[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_state: torch.Tensor):
        h = self.act(self.c_fc(hidden_state))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, config: OpenAIGPTConfig, scale=False) -> None:
        super().__init__()
        self.attn = Attention(config, scale)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, hidden_state, attention_mask=None):
        residual = hidden_state

        attn_outputs = self.attn(hidden_state, attention_mask=attention_mask)

        n = self.ln_1(residual + attn_outputs)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):

    def __init__(self, config: OpenAIGPTConfig):
        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.register_buffer('position_ids',
                             torch.arange(config.n_positions),
                             persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, :input_shape[-1]]

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(
                self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype).min

        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):

    def __init__(self, config: OpenAIGPTConfig) -> None:
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        hidden_states = self.transformer(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         position_ids=position_ids)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor,
                                      **kwargs) -> Dict[str, Any]:
        return {'input_ids': input_ids}
