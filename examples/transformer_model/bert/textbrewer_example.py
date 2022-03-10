'''
Author: jianzhnie
Date: 2022-03-10 17:11:17
LastEditTime: 2022-03-10 17:13:43
LastEditors: jianzhnie
Description:

'''
# Defined in Section 8.3.5.3

import textbrewer
import torch
from datasets import load_dataset
from textbrewer import DistillationConfig, GeneralDistiller, TrainingConfig
from torch.utils.data import Dataloader
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          DistilBertForSequenceClassification)


def encode(examples):
    return tokenizer(examples['sentence'],
                     truncation=True,
                     padding='max_length')


def collate_fn(examples):
    return dict(tokenizer.pad(examples, return_tensors='pt'))


if __name__ == '__main__':
    # 加载数据并构建Dataloader
    dataset = load_dataset('glue', 'sst2', split='train')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    dataset = dataset.map(encode, batched=True)
    encoded_dataset = dataset.map(
        lambda examples: {'labels': examples['label']}, batched=True)
    columns = ['input_ids', 'attention_mask', 'labels']
    encoded_dataset.set_format(type='torch', columns=columns)

    dataloader = Dataloader(encoded_dataset,
                            collate_fn=collate_fn,
                            batch_size=8)

    # 定义教师和学生模型
    teacher_model = BertForSequenceClassification.from_pretrained(
        'bert-base-cased')
    student_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-cased')

    # 打印教师模型和学生模型的参数量（可选）
    print("\nteacher_model's parameters:")
    result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
    print(result)

    print("student_model's parameters:")
    result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
    print(result)

    # 定义优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        teacher_model.to(device)
        student_model.to(device)

    # 定义adaptor、训练配置、蒸馏配置
    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs[1]}

    train_config = TrainingConfig(device=device)
    distill_config = DistillationConfig()

    # 定义distiller
    distiller = GeneralDistiller(train_config=train_config,
                                 distill_config=distill_config,
                                 model_T=teacher_model,
                                 model_S=student_model,
                                 adaptor_T=simple_adaptor,
                                 adaptor_S=simple_adaptor)

    # 开始蒸馏！
    with distiller:
        distiller.train(optimizer,
                        dataloader,
                        scheduler_class=None,
                        scheduler_args=None,
                        num_epochs=1,
                        callback=None)
