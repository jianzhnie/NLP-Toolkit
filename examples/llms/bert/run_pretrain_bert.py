# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
from typing import Optional

import torch
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader

sys.path.append('../../')
sys.path.append('llm/nlp-toolkit/nlptoolkit')
from nlptoolkit.datasets.bertdataset import BertDataset
from nlptoolkit.llms.bert.config_bert import BertConfig
from nlptoolkit.llms.bert.modeling_output import BertForPreTrainingOutput
from nlptoolkit.llms.bert.tasking_bert import BertForPreTraining


class MMBertForClassify(BaseModel):

    def __init__(self, model: BertForPreTraining):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        mode: str = 'loss',
    ):
        # Forward pass through BERT base model
        outputs: BertForPreTrainingOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=masked_lm_labels,
            next_sentence_label=next_sentence_label,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if mode == 'loss':
            return {
                'loss': outputs.loss,
                'masked_lm_loss': outputs.masked_lm_loss,
                'next_sentence_loss': outputs.next_sentence_loss,
            }
        elif mode == 'predict':
            return (
                outputs.loss,
                outputs.masked_lm_loss,
                outputs.next_sentence_loss,
                outputs.prediction_logits,
                outputs.seq_relationship_logits,
                masked_lm_labels,
                next_sentence_label,
            )


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        (
            loss,
            masked_lm_loss,
            next_sentence_loss,
            prediction_logits,
            seq_relationship_logits,
            masked_lm_labels,
            next_sentence_label,
        ) = data_samples

        correct = ((seq_relationship_logits.argmax(
            dim=1) == next_sentence_label).sum().cpu())
        self.results.append({
            'loss': loss,
            'masked_lm_loss': masked_lm_loss,
            'next_sentence_loss': next_sentence_loss,
            'batch_size': len(next_sentence_label),
            'correct': correct,
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        accuracy = 100 * total_correct / total_size

        outputs = {
            key: val
            for key, val in results[0].items()
            if key not in ['batch_size', 'correct']
        }
        outputs = {
            key: sum(item[key] for item in results) / len(results)
            for key in outputs
        }
        outputs['accuracy'] = accuracy
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = BertConfig()
    model = BertForPreTraining(config)
    data_dir = '/home/robin/work_dir/llm/nlp-toolkit/text_data/wikitext-2/'
    train_set = BertDataset(data_dir=data_dir,
                            data_split='valid',
                            max_seq_len=128)
    valid_set = BertDataset(data_dir=data_dir,
                            data_split='valid',
                            max_seq_len=128)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(valid_set, batch_size=32, shuffle=True)

    train_dataloader = dict(
        batch_size=32,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'),
    )
    val_dataloader = dict(
        batch_size=32,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'),
    )

    runner = Runner(
        model=MMBertForClassify(model),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optim_wrapper=dict(optimizer=dict(type=torch.optim.Adam, lr=2e-5)),
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        visualizer=dict(
            type='Visualizer',
            vis_backends=[
                dict(
                    type='WandbVisBackend',
                    init_kwargs=dict(project='bert-pretrain'),
                )
            ],
        ),
        work_dir='bert_work_dir',
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
