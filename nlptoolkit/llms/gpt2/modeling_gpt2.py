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

    def scale_dotproductt_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the Scaled Dot Product Attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output of the attention.
        """
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1)**0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

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
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim,
                                                             dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self.scale_dotproductt_attn(query, key, value,
                                                  attention_mask)

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
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)
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
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is None:
            position_ids = torch.arange(0,
                                        input_shape[-1],
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype).min

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
            hidden_states = block(hidden_states, attention_mask=attention_mask)

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
        attention_mask: Optional[torch.FloatTensor] = None,
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
            attention_mask=attention_mask,
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 5,
    ) -> torch.LongTensor:
        """Take a conditioning sequence of indices idx (LongTensor of shape
        (b,t)) and complete the sequence max_new_tokens times, feeding the
        predictions back into the model each time.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        assert max_new_tokens > 0, 'max_new_tokens must > 0'
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            input_idx_crop = (input_ids
                              if input_ids.size(1) <= self.config.n_positions
                              else input_ids[:, -self.config.n_positions:])
            # forward the model to get the logits for the index in the sequence
            outputs = self.forward(input_idx_crop)
            logits = outputs.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            input_ids_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

        return input_ids


if __name__ == '__main__':
    import torch
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    config = GPT2Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel(config=config).to(device)
    text = 'How are you?'
    inputs = tokenizer(text, return_tensors='pt')
    print(inputs)
    inputs['attention_mask'] = torch.tensor([[1, 1, 0, 0]])
    print(inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    print(outputs)
    new_ids = model.generate(input_ids=inputs['input_ids'], max_new_tokens=10)
    print(new_ids)
