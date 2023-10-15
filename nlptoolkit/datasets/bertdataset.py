'''
Author: jianzhnie
Date: 2022-03-04 17:13:34
LastEditTime: 2022-03-04 17:16:27
LastEditors: jianzhnie
Description:

'''
import os
import random
import sys
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

sys.path.append('/home/robin/work_dir/llm/nlp-toolkit')
from nlptoolkit.data.tokenizer import Tokenizer
from nlptoolkit.data.vocab import Vocab


class BertDataset(Dataset):
    """
    Dataset class for BERT pretraining tasks.

    Args:
        data_dir (str): Directory containing the data files.
        data_split (str): Split of the data ('train', 'val', etc.).
        tokenizer (Tokenizer): Tokenizer object for tokenizing the text.
        max_seq_len (int): Maximum sequence length for BERT input.
    """
    def __init__(
            self,
            data_dir: str,
            data_split: str = 'train',
            tokenizer: Tokenizer = Tokenizer(),
            max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_dir = os.path.join(data_dir, f'{data_split}.txt')
        self.paragraphs = self.preprocess_text_data(self.data_dir)
        self.tokenized_paragraphs = self.tokenize_text(self.paragraphs)
        self.vocab: Vocab = self.build_vocab()
        self.vocab_words = list(self.vocab.token_to_idx.keys())
        (self.all_token_ids, self.all_token_type_ids, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = self.get_bert_data(self.tokenized_paragraphs,
                                               max_seq_len)

    def tokenize_text(self,
                      paragraphs: List[List[str]]) -> List[List[List[str]]]:
        tokenized_paragraphs = []
        for paragraph in paragraphs:
            tokenized_paragraph = []
            for sentence in paragraph:
                tokenized_sentences = self.tokenizer.tokenize(sentence)
                tokenized_paragraph.append(tokenized_sentences)
            tokenized_paragraphs.append(tokenized_paragraph)
        return tokenized_paragraphs

    def build_vocab(self) -> Vocab:
        """
        Build vocabulary from tokenized sentences.

        Returns:
            Vocab: Vocabulary object.
        """
        tokenized_sentences = [
            sentence for paragraph in self.tokenized_paragraphs
            for sentence in paragraph
        ]
        vocab = Vocab.build_vocab(tokenized_sentences,
                                  min_freq=1,
                                  unk_token='<unk>',
                                  pad_token='<pad>',
                                  bos_token='<bos>',
                                  eos_token='<eos>',
                                  cls_token='<cls>',
                                  seq_token='<sep>')
        return vocab

    def get_bert_data(
            self, paragraphs: List[List[str]],
            max_seq_len: int) -> List[Tuple[List[str], List[int], bool]]:
        """
        Get BERT pretraining data from tokenized paragraphs.

        Args:
            paragraphs (List[List[str]]): List of tokenized paragraphs.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            List[Tuple[List[str], List[int], bool]]: List of BERT pretraining data tuples.
        """
        examples = []
        for paragraph in paragraphs:
            nsp_data_from_paragraph = self.get_nsp_data_from_paragraph(
                paragraph, paragraphs, max_seq_len)
            examples.extend(nsp_data_from_paragraph)

        # 获取遮蔽语言模型任务的数据
        bert_data = []
        for tokens, segments, is_next in examples:
            mlm_input_tokens, mlm_pred_positions, mlm_pred_labels = self.get_mlm_data_from_tokens(
                tokens)
            bert_data.append((mlm_input_tokens, mlm_pred_positions,
                              mlm_pred_labels) + (segments, is_next))

        formated_bert_data = self.format_bert_inputs(bert_data, max_seq_len)

        return formated_bert_data

    def get_next_sentence(
            self, sentence: List[str], next_sentence: List[str],
            paragraphs: List[List[List[str]]]) -> Tuple[List[str], bool]:
        """
        Randomly selects the next sentence for NSP task.

        Args:
            sentence (List[str]): Current sentence.
            next_sentence: (List[str]): Next sentence.
            paragraphs (List[List[str]]): List of paragraphs.

        Returns:
            Tuple[List[str], List[str], bool]: Tokens of next sentence and whether they are consecutive sentences.
        """
        if random.random() < 0.5:
            is_next = 1
        else:
            # paragraphs 是三重列表的嵌套
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = 0
        return sentence, next_sentence, is_next

    def get_nsp_data_from_paragraph(
            self, paragraph: List[str], paragraphs: List[List[List[str]]],
            max_seq_len: int) -> List[Tuple[List[str], bool]]:
        """
        Generate NSP (Next Sentence Prediction) data from a paragraph.

        Args:
            paragraph (List[List[str]]): Tokenized paragraph.
            paragraphs (List[List[List[str]]]): List of paragraphs.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            List[Tuple[List[str], bool]]: List of NSP data tuples.
        """
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self.get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)
            # 考虑1个'<cls>'词元和2个'<sep>'词元
            if len(tokens_a) + len(tokens_b) + 3 > max_seq_len:
                continue
            tokens, segments = self.get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
        return nsp_data_from_paragraph

    def get_tokens_and_segments(
            self,
            tokens_a: List[str],
            tokens_b: List[str] = None) -> Tuple[List[str], List[int]]:
        """
        Get tokens and segments for BERT input.

        Args:
            tokens_a (List[str]): Tokens of the sentence.

        Returns:
            Tuple[List[str], List[int]]: Tokens and corresponding segments.
        """
        tokens = ['<cls>'] + tokens_a + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b is not None:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments

    def get_mlm_data_from_tokens(
            self, tokens: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """
        Generates Masked Language Model (MLM) data from a list of tokens.

        Args:
            tokens (List[str]): List of tokens in a sentence.

        Returns:
            Tuple[List[int], List[int], List[int]]: Indices of MLM input tokens, positions of masked tokens, and MLM labels.
        """
        candidate_pred_positions = []
        for i, token in enumerate(tokens):
            # Exclude special tokens from prediction
            if token not in ['<cls>', '<sep>']:
                candidate_pred_positions.append(i)

        # Predict 15% of the tokens
        num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))
        mlm_input_tokens, mlm_pred_positions, mlm_pred_labels = self.replace_mlm_tokens(
            tokens,
            candidate_pred_positions,
            num_mlm_preds,
            vocab_words=self.vocab_words,
        )

        return mlm_input_tokens, mlm_pred_positions, mlm_pred_labels

    def replace_mlm_tokens(
            self, tokens: List[str], candidate_pred_positions: List[int],
            num_mlm_preds: int,
            vocab_words: List[str]) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Replaces tokens with <mask> or random tokens to create MLM training data.

        Args:
            tokens (List[str]): List of tokens in a sentence.
            candidate_pred_positions (List[int]): List of indices where predictions can be made.
            num_mlm_preds (int): Number of tokens to predict.
            vocab_words (List[str]): Vocab word list.

        Returns:
            Tuple[List[int], List[int], List[int]]: Indices of MLM input tokens and positions with corresponding labels.
        """
        # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
        mlm_input_tokens = tokens.copy()
        mlm_pred_positions = []
        mlm_pred_labels = []

        # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(mlm_pred_positions) >= num_mlm_preds:
                break
            masked_token = None
            # 80% of the time, replace with <mask>
            if random.random() < 0.8:
                masked_token = '<mask>'
            else:
                # 10% of the time, keep the word unchanged
                if random.random() < 0.5:
                    masked_token = mlm_input_tokens[mlm_pred_position]
                # 10% of the time, replace with a random word from vocabulary
                else:
                    masked_token = random.choice(vocab_words)

            mlm_input_tokens[mlm_pred_position] = masked_token
            mlm_pred_positions.append(mlm_pred_position)
            mlm_pred_labels.append(tokens[mlm_pred_position])

        return mlm_input_tokens, mlm_pred_positions, mlm_pred_labels

    def format_bert_inputs(
        self,
        examples: List[Tuple[List[str], List[int], List[int], List[int],
                             List[int], bool]],
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Formats BERT pretraining inputs from examples.

        Args:
            examples (List[Tuple[List[str], List[int], List[int], List[int], List[int], bool]]): List of BERT pretraining examples.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Formatted BERT pretraining inputs (tokens, segments, valid lengths, MLM prediction positions, MLM weights, MLM labels, NSP labels).
        """
        max_num_mlm_preds = round(max_seq_len * 0.15)
        all_token_ids, all_token_type_ids, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []

        for (mlm_input_tokens, mlm_pred_positions, mlm_pred_labels, segments,
             is_next) in examples:

            # Tokenize and pad tokens
            token_ids = self.vocab[mlm_input_tokens]
            token_ids += [self.vocab['<pad>']] * (max_seq_len - len(token_ids))

            all_token_ids.append(token_ids)

            # Pad segments
            token_type_ids = segments + [0] * (max_seq_len - len(segments))
            all_token_type_ids.append(token_type_ids)

            # Compute valid lengths (excluding padding tokens)
            valid_lens.append(len(mlm_input_tokens))

            # Pad MLM prediction positions, MLM weights, and MLM labels
            mlm_positions = mlm_pred_positions + [0] * (
                max_num_mlm_preds - len(mlm_pred_positions))

            mlm_weights = [1.0] * len(mlm_pred_labels) + [0.0] * (
                max_num_mlm_preds - len(mlm_pred_labels))

            mlm_labels = self.vocab[mlm_pred_labels] + [0] * (
                max_num_mlm_preds - len(mlm_pred_labels))

            all_pred_positions.append(mlm_positions)
            all_mlm_weights.append(mlm_weights)
            all_mlm_labels.append(mlm_labels)

            # NSP labels
            nsp_labels.append(is_next)

        # Convert lists to tensors and return
        return all_token_ids, all_token_type_ids, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels

    def preprocess_text_data(self, path: str) -> List[List[str]]:
        """
        Preprocesses the text data from the specified file.

        Args:
            path (str): Path to the text file.

        Returns:
            List[List[str]]: List of tokenized sentences.
        """
        assert os.path.exists(path)
        paragraphs = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.split(' . ')) >= 2:
                    paragraph = line.strip().lower().split(' . ')
                    paragraphs.append(paragraph)
        return paragraphs

    def __len__(self) -> int:
        """
        Returns the total number of paragraphs in the dataset.

        Returns:
            int: Number of paragraphs.
        """
        return len(self.all_token_ids)

    def __getitem__(self, idx: int) -> List[Tuple[List[str], List[int], bool]]:
        """
        Retrieves the NSP data for a specific paragraph.

        Args:
            idx (int): Index of the paragraph.

        Returns:
            List[Tuple[List[str], List[int], bool]]: List of NSP data tuples.
        """
        inputs = dict(
            token_ids=torch.tensor(self.all_token_ids[idx], dtype=torch.long),
            token_type_ids=torch.tensor(self.all_token_type_ids[idx],
                                        dtype=torch.long),
            valid_len=torch.tensor(self.valid_lens[idx], dtype=torch.float32),
            pred_positions=torch.tensor(self.all_pred_positions[idx],
                                        dtype=torch.long),
            mlm_weight=torch.tensor(self.all_mlm_weights[idx],
                                    dtype=torch.float32),
            labels=torch.tensor(self.all_mlm_labels[idx], dtype=torch.long),
            nsp_labels=torch.tensor(self.nsp_labels[idx], dtype=torch.long),
        )

        return inputs


if __name__ == '__main__':
    data_dir = '/home/robin/work_dir/llm/nlp-toolkit/text_data/wikitext-2/'
    bert_dataset = BertDataset(data_dir=data_dir, max_seq_len=64)
    for i in range(5):
        print(bert_dataset[i])
