from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer  # Type for tokenizers


class GPTDataset(Dataset):
    """
    A PyTorch Dataset for creating sliding window chunks of tokenized text data for GPT training.

    This dataset splits a large text into overlapping sequences using a sliding window approach,
    which is useful for training language models that need to maintain context across sequences.

    Attributes:
        input_ids (List[torch.Tensor]): List of input sequences
        target_ids (List[torch.Tensor]): List of target sequences (input shifted by 1)
    """

    def __init__(
        self,
        txt: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        """
        Initialize the GPTDataset.

        Args:
            txt (str): The input text to be processed
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text
            max_length (int): Maximum length of each sequence
            stride (int): Number of tokens to slide the window by between chunks

        Raises:
            ValueError: If max_length or stride are invalid, or if text is empty
            TypeError: If inputs are of incorrect type
        """
        # Input validation
        if not isinstance(txt, str) or not txt.strip():
            raise ValueError("Input text must be a non-empty string")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        # Tokenize the entire text
        try:
            token_ids = tokenizer.encode(
                txt, allowed_special={"<|endoftext|>"}, add_special_tokens=True
            )
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}")

        # Validate token sequence length
        if len(token_ids) <= max_length:
            raise ValueError(
                f"Text is too short after tokenization. Got {len(token_ids)} tokens, "
                f"need more than {max_length}"
            )

        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract chunks for input and target
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Convert to tensors and store
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.

        Returns:
            int: The number of sequences
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx (int): The index of the example to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (input_sequence, target_sequence)

        Raises:
            IndexError: If idx is out of bounds
        """
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        return self.input_ids[idx], self.target_ids[idx]

    def get_sequence_length(self) -> int:
        """
        Get the length of sequences in the dataset.

        Returns:
            int: The length of each sequence, or 0 if dataset is empty
        """
        if not self.input_ids:
            return 0
        return len(self.input_ids[0])
