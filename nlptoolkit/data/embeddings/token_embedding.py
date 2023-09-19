'''
Author: jianzhnie
Date: 2022-03-25 17:43:24
LastEditTime: 2022-03-25 18:57:26
LastEditors: jianzhnie
Description:

'''
import os
from typing import List, Tuple, Union

import torch


class TokenEmbedding:
    """
    TokenEmbedding class for loading and using pre-trained word embeddings like GloVe or fastText.
    """
    def __init__(self, embedding_name: str):
        """
        Initialize the TokenEmbedding instance.

        Args:
            embedding_name (str): Name of the pre-trained word embedding model to load (e.g., "glove.6B.50d").
        """
        self.idx_to_token, self.idx_to_vec = self.load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def load_embedding(self,
                       embedding_name: str) -> Tuple[List[str], torch.Tensor]:
        """
        Load the pre-trained word embeddings from a file.

        Args:
            embedding_name (str): Name of the pre-trained word embedding model to load.

        Returns:
            Tuple[List[str], torch.Tensor]: List of tokens and a tensor of word vectors.
        """
        idx_to_token, idx_to_vec = ['<unk>'], []

        # Load the word embeddings from the file
        with open(os.path.join(embedding_name, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)

        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens: Union[str, List[str]]) -> torch.Tensor:
        """
        Get word embeddings for one or more tokens.

        Args:
            tokens (str or List[str]): Token(s) for which to retrieve embeddings.

        Returns:
            torch.Tensor: Word embeddings as a tensor.
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        indices = [
            self.token_to_idx.get(token, self.unknown_idx) for token in tokens
        ]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self) -> int:
        """
        Get the size of the vocabulary, including the '<unk>' token.

        Returns:
            int: Size of the vocabulary.
        """
        return len(self.idx_to_token)


def find_k_nearest_neighbors(embedding: TokenEmbedding,
                             query_word: str,
                             k: int = 5) -> List[str]:
    """
    Find the K-nearest neighbors for a query word based on cosine similarities between word vectors.

    Args:
        embedding (TokenEmbedding): An instance of the TokenEmbedding class.
        query_word (str): The word for which to find nearest neighbors.
        k (int, optional): The number of nearest neighbors to retrieve. Default is 5.

    Returns:
        List[str]: A list of K-nearest neighbor words.
    """
    if query_word not in embedding.token_to_idx:
        return []
    # Return an empty list if the query word is not in the vocabulary

    query_vector = embedding(query_word)
    similarity_scores = torch.matmul(embedding.idx_to_vec, query_vector.T)
    similarity_scores /= (torch.norm(embedding.idx_to_vec, dim=1) *
                          torch.norm(query_vector))

    # Get the indices of the K-nearest neighbors (excluding the query word itself)
    top_k_indices = similarity_scores.argsort(dim=0, descending=True)[1:k + 1]

    # Retrieve the corresponding words
    nearest_neighbors = [embedding.idx_to_token[i] for i in top_k_indices]

    return nearest_neighbors


if __name__ == '__main__':
    # Example usage:
    embedding = TokenEmbedding('glove.6B.50d')
    query_word = 'king'
    similar_words = find_k_nearest_neighbors(embedding, query_word, k=5)
    print(f"Words similar to '{query_word}': {similar_words}")
