import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import Conv1D

from .config_gpt2 import GPT2Config


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        max_positions = config.n_positions
        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.bool)).view(1, 1, max_positions,
                                                   max_positions),
        )

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        # key, query, value projections for all heads, but in a batch
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # regularization
        self.attn_pdrop = config.attn_pdrop
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _split_heads(self, x: torch.Tensor, num_heads: int,
                     head_dim: int) -> torch.Tensor:
        """Splits hidden_size dim into head_dim and num_heads."""
        # x: (batch_size, seq_len, d_model) ->  (batch_size, seq_len, self.nhead, self.d_k)
        new_x_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(new_x_shape)
        # (batch_size, self.nhead, seq_len, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: torch.Tensor, num_heads: int,
                     head_dim: int) -> torch.Tensor:
        """Merges head_dim dim and num_attn_heads dim into hidden_size."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim, )
        return tensor.view(new_shape)

    def caculate_attn_weight(self, query: torch.Tensor,
                             key: torch.Tensor) -> torch.FloatTensor:
        """
        causal_mask:
            [True, False, False,
             True, True,  False,
             True, True,  True]


        Args:
            query (torch.Tensor): _description_
            key (torch.Tensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """
        d_k = query.size(-1)
        attn_weights = torch.matmul(query, key.transpose(-2,
                                                         -1)) / math.sqrt(d_k)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)

        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length]
        print('causal_mask:', causal_mask)
        mask_value = torch.finfo(attn_weights.dtype).min
        print('mask_value', mask_value)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device)
        print('mask_value:', mask_value)
        attn_weights = torch.where(causal_mask,
                                   attn_weights.to(attn_weights.dtype),
                                   mask_value)
        print('attn_weights:', attn_weights)
        return attn_weights

    def scale_dotproductt_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Scaled Dot Product Attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output of the attention.
        """
        d_k = query.size(-1)
        attn_weights = torch.matmul(query, key.transpose(-2,
                                                         -1)) / math.sqrt(d_k)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        # casual_mask is a upper matrix:
        #   [True, False, False,
        #    True, True,  False,
        #    True, True,  True]
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length]
        # mask_value: -inf
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device)
        # If value in causal_mask is True, fill attn_weights, else fill mask_value
        attn_weights = torch.where(causal_mask,
                                   attn_weights.to(attn_weights.dtype),
                                   mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def forward(self,
                hidden_states: Optional[torch.FloatTensor]) -> torch.Tensor:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim,
                                                             dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self.scale_dotproductt_attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class GPT2MLP(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        embed_dim = config.n_embd
        intermediate_size = (config.n_inner
                             if config.n_inner is not None else 4 * embed_dim)
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        # residual connection
        hidden_states = attn_output + residual

        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + mlp_output
        return hidden_states


class GPT2Model(GPT2PreTrainedModel):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings) -> None:
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        device = input_ids.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is None:
            position_ids = torch.arange(0,
                                        input_shape[-1],
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0)

        # token embeddings of shape (b, t, n_embd)
        inputs_embeds = self.wte(input_ids)
        # position embeddings of shape (t, n_embd)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPT2LMHeadModel(GPT2PreTrainedModel):

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        hidden_states = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(loss=loss,
                                                 logits=lm_logits,
                                                 hidden_states=hidden_states)


if __name__ == '__main__':
    import torch
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    config = GPT2Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2Model(config=config).to(device)
    text = 'Hello, my dog is cute!, How are you doing?'
    max_length = 1024
    inputs = tokenizer(text,
                       return_tensors='pt',
                       max_length=max_length,
                       truncation=True)
    print(inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(inputs['input_ids'])
    print(outputs)
