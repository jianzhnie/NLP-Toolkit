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


class GPTConfig:
    """base GPT config, params common to all GPT versions."""
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CustomSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.nhead = n_head
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
        value = self.value_linear(value).view(nbatches, -1,
                                              self.n_head, self.d_k).transpose(
                                                  1, 2).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self_attn = self.attention(query, key, value, mask=self.mask)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0,
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
        return output, self_attn


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
        src2 = self.attn(src2, src2, src2, mask=src_mask)[0]
        src = src + src2

        src2 = self.layer_norm2(src)
        src2 = self.mlp(src2)
        src = src + src2
        return src


class GPTModel(nn.Module):
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


def configure_optimizers(model, learning_rate, weight_decay, betas):
    """This long function is unfortunately doing something very simple and is
    being very defensive:

    We are separating out all parameters of the model into two buckets: those that will experience weight decay for regularization and those that won't (biases,
    and layernorm/embedding weights). We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(
                    m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(
                    m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(
        inter_params
    ) == 0, 'parameters %s made it into both decay/no_decay sets!' % (
        str(inter_params), )
    assert len(
        param_dict.keys() - union_params
    ) == 0, 'parameters %s were not separated into either decay/no_decay set!' % (
        str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {
            'params': [param_dict[pn] for pn in sorted(list(decay))],
            'weight_decay': weight_decay
        },
        {
            'params': [param_dict[pn] for pn in sorted(list(no_decay))],
            'weight_decay': 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
