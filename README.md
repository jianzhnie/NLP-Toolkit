<!--
 * @Author: jianzhnie
 * @Date: 2021-12-30 13:26:04
 * @LastEditTime: 2022-03-08 14:36:13
 * @LastEditors: jianzhnie
 * @Description:
 *
-->
# NLPToolkit

<img src="docs/imgs/PLMfamily.jpg" alt="PLMfamily" style="zoom:200%;" />

## Introduction
**`nlptoolkit`** is a toolkit for NLP(Natural Language Processing) and using **Pytorch**.  **`nlptoolkit`**  has implemented many deep learning models and data preprocessing methods. More importantly, it provides a lot of examples that can run end-to-end.


## Input Embedding

- [x] [Token Embedding]()
- [x] [WordPiece tokenization embeddings]()
- [x] [Segment embeddings]()

## Positional Encoding

- [x] [PositionalEncoding]()
- [x] [Byte Pair Encoding(BPE)]()
- [x] [Learning Encoding]()

## Support Models

Supported Language Models:
- [x] [Ngram]()
- [x] [SkipGram]()
- [x] [CoVe]()
- [x] [ELMO]()
- [x] [ULMFiT]()
- [x] [Seq2Seq | Attention Seq2Seq]()
- [x] [GPT]()
- [x] [GPT2]()
- [x] [ELECTRA]()


Supported Transformer Models:
- [x] [Transformer]()
- [x] [Bert | TinyBert | DistilBert]()
- [x] [Transformer-XL]()
- [x] [Sparse Transfromer]()
- [x] [Universal Transforme]()
- [x] [XLNet]()
- [x] [ALBERT]()
- [x] [GPT]()
- [x] [GPT2]()
- [x] [RoBERTa]()
- [x] [T5]()
- [x] [Reformer]()
- [x] [GPT-3]()
- [x] [BART]()


## Summary

|         |           Base model            |                      Pretraining Tasks                       |
| :-----: | :-----------------------------: | :----------------------------------------------------------: |
|  CoVe   |        seq2seq NMT model        |        supervised learning using translation dataset.        |
|  ELMo   |        two-layer biLSTM         |                    next token prediction                     |
|   CVT   |        two-layer biLSTM         | semi-supervised learning using both labeled and unlabeled datasets |
| ULMFiT  |            AWD-LSTM             |          autoregressive pretraining on Wikitext-103          |
|   GPT   |       Transformer decoder       |                    next token prediction                     |
|  BERT   |       Transformer encoder       |        mask language model + next sentence prediction        |
| ALBERT  | same as BERT but light-weighted |       mask language model + sentence order prediction        |
|  GPT-2  |       Transformer decoder       |                    next token prediction                     |
| RoBERTa |          same as BERT           |            mask language model (dynamic masking)             |
|   T5    |  Transformer encoder + decoder  | pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format. |
|  GPT-3  |       Transformer decoder       |                    next token prediction                     |
|  XLNet  |          same as BERT           |                permutation language modeling                 |
|  BART   |   BERT encoder + GPT decoder    |            reconstruct text from a noised version            |
| ELECTRA |          same as BERT           |                   replace token detection                    |


## Task & Datasets

[**Common Tasks and Datasets**](docs/nlp_task_datasets.md)

## Curriculum - (Example Purpose)

### 1. Basic Embedding Model

- [NNLM(Neural Network Language Model)](nlptoolkit/models/word2vec/word2vec.py) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Word2Vec(Skip-gram)](nlptoolkit/models/word2vec/word2vec.py) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)

### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**

### 4. Attention Mechanism
- 4-1. [Seq2Seq](4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**

### 5. Model based on Transformer

- 5-1.  [The Transformer](5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- 5-2. [BERT](5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)

## Dependencies

- Python 3.5+
- Pytorch 1.5.0+


## Reference:

- https://zh.d2l.ai/
    - Dive into Deep Learning，D2L.ai
- https://github.com/dmlc/gluon-nlp/
    - GluonNLP: NLP made easy
- https://github.com/huggingface/tokenizers
    - Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.
- https://github.com/The-AI-Summer/self-attention-cv
    - Self-attention building blocks for computer vision applications in PyTorch
- 《[自然语言处理：基于预训练模型的方法](https://item.jd.com/13344628.html)》（作者：车万翔、郭江、崔一鸣）
