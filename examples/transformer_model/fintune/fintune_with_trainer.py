import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)


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
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    next(iter(dataloader))

    print(train_dataset.features)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased')
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        learning_rate=3e-4,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1,
        do_train=True,
        do_eval=True,
        eval_steps=100,
        evaluation_strategy='epoch')

    trainer = Trainer(
        model=model,  # the instantiated ? Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics)

    train_out = trainer.train()
