import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


class LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """LlamaRMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_seq_len: int = 2048,
        device: str = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Build here to make `torch.jit.trace` work.
        # precompute cos_cached, sin_cached in fp32
        self._build_cache(
            base=base,
            seq_len=max_seq_len,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _build_cache(
        self,
        base: int = 10000,
        seq_len: int = 2048,
        device: str = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.dim, 2, dtype=dtype, device=device) / self.dim))
        t = torch.arange(seq_len, dtype=dtype, device=device)
        # 计算两个矩阵的外积，结果维度[seq_len, dim // 2]
        # freqs = [[0, 0, 0,..., 0],
        #          [1/10000.0^(0/128), 1/10000.0^(2/128), 1/10000.0^(4/128), ..., 1/10000.0^(126/128)],
        #          [2/10000.0^(0/128), 2/10000.0^(2/128), 2/10000.0^(4/128), ..., 2/10000.0^(126/128)],
        #          ...,
        #          [4095/10000.0^(0/128), 4095/10000.0^(2/128), 4095/10000.0^(4/128), ..., 4095/10000.0^(126/128)]]
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb.shape = [seq_len, dim]
        cos_cached = emb.cos().to(dtype)
        sin_cached = emb.sin().to(dtype)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self,
                x: torch.Tensor,
                seq_len: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 超过预设的 max_seq_len 则重新计算更大的Rope缓存，否则直接在缓存上切片
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        dim: int = 768,
        base: int = 10000,
        max_seq_len: int = 2048,
        scaling_factor: float = 1.0,
        device: str = None,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(dim, base, max_seq_len, device)

    def _build_cache(
        self,
        base: int = 10000,
        seq_len: int = 2048,
        device: str = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.dim, 2, dtype=dtype, device=device) / self.dim))
        t = torch.arange(seq_len, dtype=dtype, device=device)
        t = t / self.scaling_factor
        # 线性内插相当于将位置序号等比例缩小

        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(dtype)
        sin_cached = emb.sin().to(dtype)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        dim: int = 768,
        base: int = 10000,
        max_seq_len: int = 2048,
        scaling_factor: float = 1.0,
        device: str = None,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(dim, base, max_seq_len, device)

    def _build_cache(
        self,
        base: int,
        seq_len: int,
        device: str = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len_cached = seq_len
        # NTK扩展方式直接对base进行缩放
        if seq_len > self.max_seq_len:
            base = base * ((self.scaling_factor * seq_len / self.max_seq_len) -
                           (self.scaling_factor - 1))**(self.dim /
                                                        (self.dim - 2))
            inv_freq = 1.0 / (base**(torch.arange(
                0, self.dim, 2, dtype=dtype, device=device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(seq_len, dtype=dtype, device=device)
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(dtype)
        sin_cached = emb.sin().to(dtype)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_seq_len = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).')

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self) -> None:
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                base=self.rope_theta,
                max_seq_len=self.max_seq_len,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.rope_theta,
                    max_seq_len=self.max_seq_len,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'dynamic':
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.rope_theta,
                    max_seq_len=self.max_seq_len,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (tensor.view(bsz, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2).contiguous())

    def _split_heads(self, x: torch.Tensor, num_heads: int,
                     head_dim: int) -> torch.Tensor:
        """Splits hidden_size dim into head_dim and num_heads."""
        # x: (batch_size, seq_len, d_model) ->  (batch_size, seq_len, self.nhead, self.d_k)
        new_x_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(new_x_shape)
        # (batch_size, self.nhead, seq_len, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor, num_heads: int,
                     head_dim: int) -> torch.Tensor:
        """Merges head_dim dim and num_attn_heads dim into hidden_size."""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (num_heads * head_dim, )
        return x.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states, self.num_heads,
                                         self.head_dim)
        key_states = self._split_heads(key_states, self.num_key_value_heads,
                                       self.head_dim)
        value_states = self._split_heads(value_states,
                                         self.num_key_value_heads,
                                         self.head_dim)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}')

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attention_dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,
                                   self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   self.hidden_size,
                                   bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(LlamaPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # retrieve input_ids and inputs_embeds
        batch_size, seq_length = input_ids.shape[:2]
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0,
                                        seq_length,
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None
                                            and 0 in attention_mask) else None
        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


if __name__ == '__main__':
    x = torch.randn(1, 8, 4, 2)
    rope = LlamaRotaryEmbedding(dim=8, base=10, max_seq_len=8)
    rope2 = LlamaLinearScalingRotaryEmbedding(dim=8,
                                              base=10,
                                              max_seq_len=8,
                                              scaling_factor=2)
    cos, sin = rope2.forward(x, seq_len=4)
    print(cos.shape)
    print(cos)
