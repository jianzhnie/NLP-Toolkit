# Word Embedding with NLP Toolkit

## 简介

以下通过基于开源情感倾向分类数据集ChnSentiCorp的文本分类训练例子展示 Word Embedding 的训练过程。

## 启动训练

我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在验证集（dev.tsv）验证。训练时会自动下载词表dict.txt，用于对数据集进行切分，构造数据样本。

启动训练：

```shell
python train_bowcls.py \
                --lr=5e-4 \
                --batch_size=64 \
                --epochs=20 \
                --vocab_path /home/robin/work_dir/llm/nlp-toolkit/data/word_vocab.txt

```

以上参数表示：

- `lr`: 学习率， 默认为5e-4。
- `batch_size`: 运行一个batch大小，默认为64。
- `epochs`: 训练轮次，默认为5。
- `vocab_path`: 需要加载的单词词表文件路径

## 参考论文

- Li, Shen, et al. "Analogical reasoning on chinese morphological and semantic relations." arXiv preprint arXiv:1805.06504 (2018).
- Qiu, Yuanyuan, et al. "Revisiting correlations between intrinsic and extrinsic evaluations of word embeddings." Chinese Computational Linguistics and Natural Language Processing Based on Naturally Annotated Big Data. Springer, Cham, 2018. 209-221.
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
- T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations

# Word-level Language Modeling using RNN and Transformer

This example trains a multi-layer RNN (Elman, GRU, or LSTM) or Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python train_transformer.py --cuda --epochs 6 --model Transformer --lr 5
                                           # Train a Transformer model on Wikitext-2 with CUDA.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) or Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```shell
PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --d_model D_MODEL     size of word embeddings
  --num_hidden_size NUM_HIDDEN_SIZE
                        number of hidden units per layer
  --num_layers NUM_LAYERS
                        number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --num_heads NUM_HEADS
                        the number of heads in the encoder of the transformer model
```
