import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../../')
from torch.utils.data.dataset import random_split

from nlptoolkit.datasets.nmtdataset import NMTDataset
from nlptoolkit.losses import MaskedSoftmaxCELoss
from nlptoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq
from nlptoolkit.utils.model_utils import count_parameters
from nlptoolkit.utils.trainer import train_and_evaluate

if __name__ == '__main__':
    # Check for available GPU, and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Define file paths and model paths
    file_path = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    model_path = '/home/robin/work_dir/llm/nlp-toolkit/nmt/best_model.pth'

    # Create an instance of NMTDataset
    nmtdataset = NMTDataset(file_path=file_path, max_seq_len=30)
    src_vocab = nmtdataset.src_vocab
    tgt_vocab = nmtdataset.tgt_vocab

    # Split the dataset into training and validation sets
    data_train, data_val = random_split(nmtdataset, [0.7, 0.3])

    # Define batch size for DataLoader
    batch_size = 128

    # Create DataLoader for training and validation sets
    train_iter = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(data_val, batch_size=batch_size, shuffle=True)

    # Initialize the seq2seq model
    model = RNNSeq2Seq(src_vocab_size=len(src_vocab),
                       trg_vocab_size=len(tgt_vocab),
                       embed_size=32,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.5).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Initialize the loss criterion
    criterion = MaskedSoftmaxCELoss()

    train_and_evaluate(model,
                       train_iter,
                       valid_iter,
                       optimizer,
                       criterion,
                       num_epochs=10,
                       clip=0.25,
                       log_interval=10,
                       save_model_path=model_path)
