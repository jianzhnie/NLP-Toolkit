import copy
import logging
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        # key, query, value projections for all heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        # regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # output projection
        self.output_linear = nn.Linear(d_model, d_model)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = self.generate_square_subsequent_mask(block_size)
        self.n_head = n_head

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.tril(torch.ones(sz, sz)) == 1)
        mask = mask.view(1, 1, sz, sz)
        return mask

    def forward(self, query, key, value, mask=None):
        nbatches, seq_len, d_k = query.size()
        if mask is not None:
            self.mask = mask
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query = self.query_linear(query).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # key = self.key_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)
        # value = self.value_linear(key).view(nbatches, -1, self.h, self.nhead, self.d_k).transpose(1,2)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query = self.query_linear(query).view(nbatches, -1, self.n_head,
                                              self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(nbatches, -1, self.n_head,
                                        self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(nbatches, -1, self.n_head,
                                              self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0,
                                    float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout1(p_attn)
        x = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.n_head * self.d_k)
        # 4) linear proj output
        x = self.output_linear(x)
        output = self.dropout2(x)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.attn = CustomSelfAttention(d_model,
                                        n_head,
                                        block_size,
                                        dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(),
        )

    def forward(self, src, src_mask=None):
        src2 = self.layer_norm1(src)
        src2 = self.attn(src2, src2, src2, mask=src_mask)
        src = src + src2

        src2 = self.layer_norm2(src)
        src2 = self.mlp(src2)
        src = src + src2
        return src


class GPTModel(nn.Module):
    """the full GPT language model, with a context size of block_size."""
    def __init__(self,
                 vocab_size,
                 d_model,
                 n_head,
                 num_layers,
                 block_size,
                 dropout=0.1):
        super().__init__()

        # input embedding stem
        self.token_embedder = nn.Embedding(vocab_size, d_model)
        self.pos_embedder = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.dropout = nn.Dropout(dropout)

        # transformers
        self.decoder_layer = DecoderLayer(d_model,
                                          n_head,
                                          block_size,
                                          dropout=dropout)
        self.decoder = _get_clones(self.decoder_layer, num_layers)

        # decoder head
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.block_size = block_size
        self.apply(self._init_weights)
        logger.info('number of parameters: %e',
                    sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, tokens, targets=None):
        batch_size, seq_len = tokens.size()
        assert seq_len <= self.block_size, 'Cannot forward, model block size is exhausted.'

        # forward the GPT model
        # each src maps to a (learnable) vector
        token_embeddings = self.token_embedder(tokens)
        position_embeddings = self.pos_embedder[:, :seq_len, :]
        # each position maps to a (learnable) vector
        x = self.dropout(token_embeddings + position_embeddings)
        x = self.decoder(x)
        x = self.layer_norm(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        return logits, loss
