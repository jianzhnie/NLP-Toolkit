import torch
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          get_scheduler)


def tokenize_function(examples):
    tokenized = tokenizer(examples['text'],
                          padding='max_length',
                          truncation=True,
                          max_length=256)
    return tokenized


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels,
                                                               preds,
                                                               average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    # 加载数据集
    root_dir = 'data/aclImdb/'
    imdb_dataset = load_dataset('imdb')
    print(imdb_dataset)
    print('Length of training set: ', len(imdb_dataset))
    print('First example from the dataset: \n')
    print(imdb_dataset['train'][:2])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    imdb_dataset = imdb_dataset.map(
        lambda examples: {'labels': examples['label']}, batched=True)
    tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(
        type='torch',
        columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(
        range(100))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(
        range(100))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=2)
    next(iter(train_dataloader))

    model = BertForSequenceClassification.from_pretrained('bert-base-cased')

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    metric = load_metric('accuracy')

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])

    metric.compute()
