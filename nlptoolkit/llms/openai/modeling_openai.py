import math

import torch
from torch import nn
from transformers.pytorch_utils import (Conv1D,
                                        find_pruneable_heads_and_indices,
                                        prune_conv1d_layer)


class Attention(nn.Module):

    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(
                f'Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}'
            )
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(n_positions,
                                  n_positions)).view(1, 1, n_positions,
                                                     n_positions),
            persistent=False,
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head,
            self.pruned_heads)
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head -
                                                              len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self,
              q,
              k,
              v,
              attention_mask=None,
              head_mask=None,
              output_attentions=False) -> list[torch.Tensor]:
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(
            *new_x_shape)  # in Tensorflow implementation: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(
            *new_x_shape)  # in Tensorflow implementation: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self,
                x: torch.tensor,
                attention_mask=None,
                head_mask=None,
                output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask,
                                  output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)
