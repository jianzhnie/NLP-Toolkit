import tensorflow as tf
from datasets import load_dataset
from transformers import (BertTokenizerFast, DefaultDataCollator,
                          TFAutoModelForSequenceClassification)


def tokenize_function(examples):
    tokenized = tokenizer(examples['text'],
                          padding='max_length',
                          truncation=True,
                          max_length=256)
    return tokenized


if __name__ == '__main__':
    # 加载数据集
    root_dir = 'data/aclImdb/'
    imdb_dataset = load_dataset('imdb')
    print(imdb_dataset)
    print('Length of training set: ', len(imdb_dataset))
    print('First example from the dataset: \n')
    print(imdb_dataset['train'][0])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    imdb_dataset = imdb_dataset.map(
        lambda examples: {'labels': examples['label']}, batched=True)
    tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(
        range(100))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(
        range(100))

    data_collator = DefaultDataCollator(return_tensors='tf')

    tf_train_dataset = train_dataset.to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'token_type_ids'],
        label_cols=['labels'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf_validation_dataset = eval_dataset.to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'token_type_ids'],
        label_cols=['labels'],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )
    print(train_dataset.features)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased', num_labels=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(tf_train_dataset,
              validation_data=tf_validation_dataset,
              epochs=3)
