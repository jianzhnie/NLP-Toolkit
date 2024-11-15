import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../../')
from torch.utils.data.dataset import random_split

from llmtoolkit.datasets.nmtdataset import NMTDataset
from llmtoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq
from llmtoolkit.utils.logger_utils import get_outdir
from llmtoolkit.utils.trainer import predict_seq2seq

if __name__ == '__main__':
    # Check for available GPU, and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Define file paths and model paths
    file_path = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    work_dir = '/home/robin/work_dir/llm/nlp-toolkit/work_dirs'
    model_path = get_outdir(work_dir, 'rnn_mt', inc=True)

    # Create an instance of NMTDataset
    nmtdataset = NMTDataset(file_path=file_path, max_seq_len=30)
    src_vocab = nmtdataset.src_vocab
    tgt_vocab = nmtdataset.tgt_vocab
    tokenizer = nmtdataset.tokenizer

    # Split the dataset into training and validation sets
    data_train, data_val = random_split(nmtdataset, [0.99, 0.01])

    # Define batch size for DataLoader
    batch_size = 128

    # Create DataLoader for training and validation sets
    train_iter = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(data_val, batch_size=batch_size, shuffle=True)

    # Initialize the seq2seq model
    model = RNNSeq2Seq(src_vocab_size=len(src_vocab),
                       trg_vocab_size=len(tgt_vocab),
                       embed_size=32,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.5).to(device)

    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation = predict_seq2seq(model,
                                      tokenizer,
                                      eng,
                                      src_vocab,
                                      tgt_vocab,
                                      max_seq_len=30,
                                      device=device)
        print(translation)
