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
        return_dict: Optional[bool] = None,
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
                outputs.prediction_logits,
                outputs.seq_relationship_logits,
                masked_lm_labels,
                next_sentence_label,
            )


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


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
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    runner = Runner(
        model=MMBertForClassify(model),
        train_dataloader=train_loader,
        optim_wrapper=dict(optimizer=dict(type=torch.optim.Adam, lr=2e-5)),
        train_cfg=dict(by_epoch=True, max_epochs=2),
        work_dir='bert_work_dir',
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
