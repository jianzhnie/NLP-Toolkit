import argparse
import collections
import os
import random
from typing import List, NamedTuple, Tuple

import h5py
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

from nlptoolkit.llms.bert.tokenization_bert import convert_to_unicode


# Define a named tuple called MaskedLmInstance
class MaskedLmInstance(NamedTuple):
    """Represents a masked language model instance with index and label.

    Attributes:
        index (int): Index of the masked token.
        label (str): Label for the masked token, typically "[MASK]".
    """
    index: int
    label: str


class TrainingInstance:
    """A single training instance (sentence pair) for masked language model
    training.

    Attributes:
        tokens (List[str]): List of tokens representing the input sentence pair.
        segment_ids (List[int]): List of segment IDs indicating token belongingness to sentences.
        masked_lm_positions (List[int]): List of positions where tokens are masked during training.
        masked_lm_labels (List[str]): List of masked tokens corresponding to masked positions.
        is_random_next (bool): Indicates if the next sentence is randomly chosen.
    """

    def __init__(
        self,
        tokens: List[str],
        segment_ids: List[int],
        is_random_next: bool,
        masked_lm_positions: List[int],
        masked_lm_labels: List[str],
    ):
        """Initializes a TrainingInstance object.

        Args:
            tokens (List[str]): List of tokens representing the input sentence pair.
            segment_ids (List[int]): List of segment IDs indicating token belongingness to sentences.
            is_random_next (bool): Indicates if the next sentence is randomly chosen.
            masked_lm_positions (List[int]): List of positions where tokens are masked during training.
            masked_lm_labels (List[str]): List of masked tokens corresponding to masked positions.
        """
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self) -> str:
        """Returns a string representation of the TrainingInstance object.

        Returns:
            str: String representation of the TrainingInstance.
        """
        tokens_str = ' '.join(self.tokens)
        segment_ids_str = ' '.join(map(str, self.segment_ids))
        masked_positions_str = ' '.join(map(str, self.masked_lm_positions))
        masked_labels_str = ' '.join(self.masked_lm_labels)

        outputs = ''
        outputs += f'Tokens: {tokens_str}\n'
        outputs += f'Segment IDs: {segment_ids_str}\n'
        outputs += f'Masked Positions: {masked_positions_str}\n'
        outputs += f'Masked Labels: {masked_labels_str}\n'
        outputs += f'Is Random Next: {self.is_random_next}'

        return outputs

    def __repr__(self) -> str:
        """Returns a string representation of the TrainingInstance object.

        Returns:
            str: String representation of the TrainingInstance.
        """
        return self.__str__()


def write_instance_to_example_file(
    instances: List[TrainingInstance],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    max_predictions_per_seq: int,
    output_file: str,
) -> None:
    """Create TF example files from `TrainingInstance`s and save them in HDF5
    format.

    Args:
        instances (List[TrainingInstance]): List of TrainingInstance objects.
        tokenizer (PreTrainedTokenizer): Tokenizer object from the transformers library.
        max_seq_length (int): Maximum sequence length for the input.
        max_predictions_per_seq (int): Maximum number of masked LM predictions per sequence.
        output_file (str): Path to save the HDF5 output file.

    Returns:
        None
    """
    total_written = 0
    features = collections.OrderedDict()

    num_instances = len(instances)
    features['input_ids'] = np.zeros([num_instances, max_seq_length],
                                     dtype='int32')
    features['input_mask'] = np.zeros([num_instances, max_seq_length],
                                      dtype='int32')
    features['segment_ids'] = np.zeros([num_instances, max_seq_length],
                                       dtype='int32')
    features['masked_lm_positions'] = np.zeros(
        [num_instances, max_predictions_per_seq], dtype='int32')
    features['masked_lm_ids'] = np.zeros(
        [num_instances, max_predictions_per_seq], dtype='int32')
    features['next_sentence_labels'] = np.zeros(num_instances, dtype='int32')

    for inst_index, instance in enumerate(
            tqdm(instances, desc='Processing instances')):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(
            instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features['input_ids'][inst_index] = input_ids
        features['input_mask'][inst_index] = input_mask
        features['segment_ids'][inst_index] = segment_ids
        features['masked_lm_positions'][inst_index] = masked_lm_positions
        features['masked_lm_ids'][inst_index] = masked_lm_ids
        features['next_sentence_labels'][inst_index] = next_sentence_label

        total_written += 1

    print(f'Total instances processed: {total_written}')
    print('Saving data to HDF5 file...')
    f = h5py.File(output_file, 'w')
    f.create_dataset('input_ids',
                     data=features['input_ids'],
                     dtype='i4',
                     compression='gzip')
    f.create_dataset('input_mask',
                     data=features['input_mask'],
                     dtype='i1',
                     compression='gzip')
    f.create_dataset('segment_ids',
                     data=features['segment_ids'],
                     dtype='i1',
                     compression='gzip')
    f.create_dataset('masked_lm_positions',
                     data=features['masked_lm_positions'],
                     dtype='i4',
                     compression='gzip')
    f.create_dataset('masked_lm_ids',
                     data=features['masked_lm_ids'],
                     dtype='i4',
                     compression='gzip')
    f.create_dataset('next_sentence_labels',
                     data=features['next_sentence_labels'],
                     dtype='i1',
                     compression='gzip')
    f.flush()
    f.close()


def create_training_instances(
    input_files: List[str],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    dupe_factor: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    random_generator: random.Random,
) -> List[TrainingInstance]:
    """Create `TrainingInstance`s from raw text.

    Args:
        input_files (List[str]): List of input file paths containing raw text data.
        tokenizer (Any): Tokenizer object for tokenizing the input text.
        max_seq_length (int): Maximum sequence length.
        dupe_factor (int): Number of times to duplicate the input data for varied instances.
        short_seq_prob (float): Probability of using shorter sequences.
        masked_lm_prob (float): Probability of masking tokens.
        max_predictions_per_seq (int): Maximum number of masked tokens in a sequence.
        random_generator (random.Random): Random number generator for reproducibility.

    Returns:
        List['TrainingInstance']: List of TrainingInstance objects.
    """
    # Initialize a list to store tokenized documents
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    # Read raw text from input files and tokenize them
    for input_file in input_files:
        print('Creating instances from {}'.format(input_file))
        with open(input_file, 'r', encoding='utf-8') as reader:
            while True:
                line = convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    random_generator.shuffle(all_documents)

    # Get vocabulary words from the tokenizer
    vocab_words = list(tokenizer.vocab.keys())

    # Generate instances by duplicating and shuffling the documents
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(all_documents, document_index,
                                               max_seq_length, short_seq_prob,
                                               masked_lm_prob,
                                               max_predictions_per_seq,
                                               vocab_words, random_generator))

    # Shuffle the instances for randomness
    random_generator.shuffle(instances)
    return instances


def create_instances_from_document(
    all_documents: List[List[List[str]]],
    document_index: int,
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: List[str],
    random_generator: random.Random,
) -> List[TrainingInstance]:
    """Creates `TrainingInstance`s for a single document.

    Args:
        all_documents (List[List[List[str]]]): List of documents where each document is a list of segments (lists of tokens).
        document_index (int): Index of the document to process.
        max_seq_length (int): Maximum sequence length.
        short_seq_prob (float): Probability of using shorter sequences.
        masked_lm_prob (float): Probability of masking tokens.
        max_predictions_per_seq (int): Maximum number of masked tokens in a sequence.
        vocab_words (List[str]): Vocabulary words.
        random_generator (Any): Random number generator for reproducibility.

    Returns:
        List[TrainingInstance]: List of TrainingInstance objects.
    """
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.

    # Determine target sequence length with some randomness
    target_seq_length = max_num_tokens
    if random_generator.random() < short_seq_prob:
        target_seq_length = random_generator.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.

    instances = []
    current_chunk = []
    current_length = 0
    i = 0

    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)

        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # Determine how many segments from `current_chunk` go into the `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random_generator.randint(1, len(current_chunk) - 1)

                tokens_a = [
                    token for segment in current_chunk[:a_end]
                    for token in segment
                ]
                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random_generator.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random_generator.randint(
                            0,
                            len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    # If picked random document is the same as the current document
                    if random_document_index == document_index:
                        is_random_next = False

                    random_document = all_documents[random_document_index]
                    random_start = random_generator.randint(
                        0,
                        len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens,
                                  random_generator)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # Combine tokens from tokens_a and tokens_b into a single list of tokens
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'
                                                 ] + tokens_b + ['[SEP]']
                segment_ids = [0] * (len(tokens_a) +
                                     2) + [1] * (len(tokens_b) + 1)

                output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens=tokens,
                    masked_lm_prob=masked_lm_prob,
                    max_predictions_per_seq=max_predictions_per_seq,
                    vocab_words=vocab_words,
                    random_generator=random_generator,
                )

                instance = TrainingInstance(
                    tokens=output_tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_masked_lm_predictions(
    tokens: List[str],
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: List[str],
    random_generator: random.Random,
) -> Tuple[List[str], List[int], List[str]]:
    """Creates predictions for the masked LM objective.

    Args:
        tokens (List[str]): List of input tokens.
        masked_lm_prob (float): Probability of masking tokens.
        max_predictions_per_seq (int): Maximum number of masked tokens in the sequence.
        vocab_words (List[str]): Vocabulary words.
        random_generator (Any): Random number generator for reproducibility.

    Returns:
        Tuple[List[str], List[int], List[str]]: Output tokens, masked LM positions, and masked LM labels.
    """
    # Get candidate indices (excluding [CLS] and [SEP])
    cand_indexes = [
        i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]']
    ]
    random_generator.shuffle(cand_indexes)

    # Calculate the number of tokens to predict
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_lms = []
    covered_indexes = set()

    output_tokens = list(tokens)

    # Iterate through candidate indices and mask tokens
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if random_generator.random() < 0.8:
            masked_token = '[MASK]'
        else:
            # 10% of the time, keep original
            if random_generator.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word from vocabulary
            else:
                masked_token = random.choice(vocab_words)

        # Update the output tokens and create MaskedLmInstance
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=masked_token))

    # Sort masked_lms by index for consistency
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # Extract masked LM positions and labels
    masked_lm_positions = [p.index for p in masked_lms]
    masked_lm_labels = [p.label for p in masked_lms]

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(
    tokens_a: List[str],
    tokens_b: List[str],
    max_num_tokens: int,
    random_generator: random.Random,
) -> None:
    """Truncates a pair of sequences to a maximum sequence length.

    Args:
        tokens_a (List[str]): List of tokens from the first sequence.
        tokens_b (List[str]): List of tokens from the second sequence.
        max_num_tokens (int): Maximum sequence length.
        random_generator (random.Random): Random number generator for reproducibility.

    Returns:
        None
    """
    while len(tokens_a) + len(tokens_b) > max_num_tokens:
        # Determine which sequence to truncate from (front or back) for more randomness
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # Randomly truncate from the front or the back
        if random_generator.random() < 0.5:
            del trunc_tokens[0]  # Truncate from the front
        else:
            trunc_tokens.pop()  # Truncate from the back


def pasre_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--vocab_file',
                        default=None,
                        type=str,
                        required=True,
                        help='The vocabulary the BERT model will train on.')
    parser.add_argument(
        '--input_file',
        default=None,
        type=str,
        required=True,
        help=
        'The input train corpus. can be directory with .txt files or a path to a single file'
    )
    parser.add_argument(
        '--output_file',
        default=None,
        type=str,
        required=True,
        help='The output file where the model checkpoints will be written.')

    # Other parameters
    # int
    parser.add_argument(
        '--max_seq_length',
        default=128,
        type=int,
        help=
        'The maximum total input sequence length after WordPiece tokenization. \n'
        'Sequences longer than this will be truncated, and sequences shorter \n'
        'than this will be padded.')
    parser.add_argument(
        '--dupe_factor',
        default=10,
        type=int,
        help=
        'Number of times to duplicate the input data (with different masks).')
    parser.add_argument('--max_predictions_per_seq',
                        default=20,
                        type=int,
                        help='Maximum sequence length.')

    # floats
    parser.add_argument('--masked_lm_prob',
                        default=0.15,
                        type=float,
                        help='Masked LM probability.')

    parser.add_argument(
        '--short_seq_prob',
        default=0.1,
        type=float,
        help=
        'Probability to create a sequence shorter than maximum sequence length'
    )

    parser.add_argument(
        '--do_lower_case',
        action='store_true',
        default=True,
        help=
        'Whether to lower case the input text. True for uncased models, False for cased models.'
    )
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help='random seed for initialization')

    args = parser.parse_args()

    return args


def main():
    args = pasre_args()
    tokenizer = BertTokenizer(args.vocab_file,
                              do_lower_case=args.do_lower_case,
                              max_len=512)

    input_files = []
    if os.path.isfile(args.input_file):
        input_files.append(args.input_file)
    elif os.path.isdir(args.input_file):
        input_files = [
            os.path.join(args.input_file, f)
            for f in os.listdir(args.input_file)
            if (os.path.isfile(os.path.join(args.input_file, f))
                and f.endswith('.txt'))
        ]
    else:
        raise ValueError('{} is not a valid path'.format(args.input_file))

    rng = random.Random(args.random_seed)
    instances = create_training_instances(input_files, tokenizer,
                                          args.max_seq_length,
                                          args.dupe_factor,
                                          args.short_seq_prob,
                                          args.masked_lm_prob,
                                          args.max_predictions_per_seq, rng)

    output_file = args.output_file

    write_instance_to_example_file(instances, tokenizer, args.max_seq_length,
                                   args.max_predictions_per_seq, output_file)


if __name__ == '__main__':
    main()
