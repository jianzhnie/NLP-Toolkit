# LLMToolkit

<img src="docs/imgs/PLMfamily.jpg" alt="PLMfamily" style="zoom:200%;" />

## Introduction

**`llmtoolkit`** is a toolkit for NLP(Natural Language Processing) and LLM(Large Language Models) using **Pytorch**.  **`llmtoolkit`**  has implemented many language models and data preprocessing methods. More importantly, it provides a lot of examples that can run end-to-end.

## Tokenizer

- [x] [BaseTokenizer](<>)
- [x] [JiebaTokenizer](<>)
- [x] [SentencePieceTokenizer](<>)
- [x] [BytePairEncoding(BPE)Tokenizer](<>)
- [x] [BertTokenizer](<>)

## Support Models

Supported Language Models:

- [x] [RNNLM](<>)
- [x] [CNNLM](<>)
- [x] [Ngram](<>)
- [x] [SkipGram](<>)
- [x] [CBOW](<>)
- [x] [Glove](<>)
- [x] [CoVe](<>)
- [x] [ELMO](<>)
- [x] [ULMFiT](<>)
- [x] [Seq2Seq | Attention Seq2Seq](<>)

Supported Transformer Models:

- [x] [Transformer](<>)
- [x] [Bert](<>)
- [x] [XLNet](<>)
- [x] [GPT](<>)
- [x] [GPT2](<>)
- [x] [RoBERTa](<>)
- [x] [T5](<>)

## Dependencies

- Python 3.7+
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
- [自然语言处理：基于预训练模型的方法](https://item.jd.com/13344628.html)（作者：车万翔、郭江、崔一鸣）

## License

`llmtoolkit` is released under the Apache 2.0 license.

## Citation

Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{llmtoolkit,
  author = {jianzhnie},
  title = {llmtoolkit: llmtoolkit is a toolkit for NLP and LLMs using Pytorch},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/LLMToolkit}},
}
```
