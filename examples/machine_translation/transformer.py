import sys
from timeit import default_timer as timer

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nlptoolkit.datasets.nmtdataset import NMTDatasets
from nlptoolkit.transformers.transformer.torch_transformer import \
    Seq2SeqTransformer

sys.path.append('../../')


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), )) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), ).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    losses = 0

    for src, tgt in train_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    losses = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_loader)


if __name__ == '__main__':
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = '/Users/jianzhengnie/work_dir/code_gallery/nlp-toolkit/examples/data'
    nmtdataset = NMTDatasets(root=root)
    src_tokens, tgt_tokens, src_vocab, tgt_vocab = nmtdataset._build_tokens()

    def generate_batch(data_batch, vocab=src_vocab):
        PAD_IDX = vocab['<pad>']
        BOS_IDX = vocab['<bos>']
        EOS_IDX = vocab['<eos>']
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), src_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
            tgt_batch.append(
                torch.cat([
                    torch.tensor([BOS_IDX]), tgt_item,
                    torch.tensor([EOS_IDX])
                ],
                          dim=0))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def get_dataloader(train_data, val_data, batch_size=128):

        train_iter = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=generate_batch)
        valid_iter = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=generate_batch)
        return train_iter, valid_iter

    data_train = nmtdataset.get_tensor_dataset(src_tokens,
                                               tgt_tokens,
                                               train=True)
    data_val = nmtdataset.get_tensor_dataset(src_tokens,
                                             tgt_tokens,
                                             train=False)

    train_iter, valid_iter = get_dataloader(data_train,
                                            data_val,
                                            batch_size=128)

    transformer = Seq2SeqTransformer(src_vocab_size=len(src_vocab),
                                     tgt_vocab_size=len(tgt_vocab),
                                     num_encoder_layers=1,
                                     num_decoder_layers=1,
                                     emb_size=32,
                                     nhead=1,
                                     dim_feedforward=32)

    transformer = transformer.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=0.0001,
                                 betas=(0.9, 0.98),
                                 eps=1e-9)

    for epoch in range(1, 10 + 1):
        start_time = timer()
        train_loss = train_epoch(transformer,
                                 train_iter,
                                 optimizer,
                                 device=device)
        end_time = timer()
        val_loss = evaluate(transformer, valid_iter, device=device)
        print((
            f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, '
            f'Epoch time = {(end_time - start_time):.3f}s'))
