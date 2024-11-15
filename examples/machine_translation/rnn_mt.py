import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../../')
from torch.utils.data.dataset import random_split

from llmtoolkit.datasets.nmtdataset import NMTDataset
from llmtoolkit.losses import MaskedSoftmaxCELoss
from llmtoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq
from llmtoolkit.utils.logger_utils import get_outdir
from llmtoolkit.utils.model_utils import count_parameters
from llmtoolkit.utils.trainer import train_and_evaluate

if __name__ == '__main__':
    # Check for available GPU, and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Define file paths and model paths
    file_path = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    work_dir = '/home/robin/work_dir/llm/nlp-toolkit/work_dirs'
    model_path = get_outdir(work_dir, 'rnn_mt', inc=True)

    embed_size = 32
    hiddens_size = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 64
    max_seq_len = 10
    learning_rate = 0.005
    num_epochs = 300
    clip_ratio = 1.0

    # Create an instance of NMTDataset
    nmtdataset = NMTDataset(file_path=file_path, max_seq_len=max_seq_len)
    src_vocab = nmtdataset.src_vocab
    tgt_vocab = nmtdataset.tgt_vocab

    # Split the dataset into training and validation sets
    data_train, data_val = random_split(
        nmtdataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    # Create DataLoader for training and validation sets
    train_iter = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    # Initialize the seq2seq model
    model = RNNSeq2Seq(src_vocab_size=len(src_vocab),
                       trg_vocab_size=len(tgt_vocab),
                       embed_size=embed_size,
                       hidden_size=hiddens_size,
                       num_layers=num_layers,
                       dropout=dropout).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Initialize the loss criterion
    criterion = MaskedSoftmaxCELoss()

    train_and_evaluate(model,
                       train_iter,
                       valid_iter,
                       optimizer,
                       criterion,
                       num_epochs=num_epochs,
                       clip_ratio=clip_ratio,
                       device=device,
                       log_interval=10,
                       save_model_path=model_path)
