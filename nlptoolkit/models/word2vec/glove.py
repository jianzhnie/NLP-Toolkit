'''
Author: jianzhnie
Date: 2021-12-23 16:23:02
LastEditTime: 2022-01-05 16:26:43
LastEditors: jianzhnie
Description:

'''
import os

import d2l.torch as d2l
import torch
import torch.nn as nn

d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        # 词嵌入及偏置向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文嵌入及偏置向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)
        return w_embeds, w_biases

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return c_embeds, c_biases

    def forward(self, words, contexts):

        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)

        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return w_embeds, w_biases, c_embeds, c_biases

    def init_weights(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, a=-0.1, b=0.1)


class GloveTokenEmbedding(object):
    """Token Embedding."""
    def __init__(self, embedding_name):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx) for token in tokens
        ]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
