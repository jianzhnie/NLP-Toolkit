## Common Tasks and Datasets

**Question-Answering**

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset): A reading comprehension dataset, consisting of questions posed on a set of Wikipedia articles, where the answer to every question is a span of text.
- [RACE](http://www.qizhexie.com/data/RACE_leaderboard) (ReAding Comprehension from Examinations): A large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students.
- See [more QA datasets in a later post](https://lilianweng.github.io/lil-log/2020/10/29/open-domain-question-answering.html#appendix-qa-datasets).

**Commonsense Reasoning**

- [Story Cloze Test](http://cs.rochester.edu/nlp/rocstories/): A commonsense reasoning framework for evaluating story understanding and generation. The test requires a system to choose the correct ending to multi-sentence stories from two options.
- [SWAG](https://rowanzellers.com/swag/) (Situations With Adversarial Generations): multiple choices; contains 113k sentence-pair completion examples that evaluate grounded common-sense inference

**Natural Language Inference (NLI)**: also known as **Text Entailment**, an exercise to discern in logic whether one sentence can be inferred from another.

- [RTE](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool) (Recognizing Textual Entailment): A set of datasets initiated by text entailment challenges.
- [SNLI](https://nlp.stanford.edu/projects/snli/) (Stanford Natural Language Inference): A collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels `entailment`, `contradiction`, and `neutral`.
- [MNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-Genre NLI): Similar to SNLI, but with a more diverse variety of text styles and topics, collected from transcribed speech, popular fiction, and government reports.
- [QNLI](https://gluebenchmark.com/tasks) (Question NLI): Converted from SQuAD dataset to be a binary classification task over pairs of (question, sentence).
- [SciTail](http://data.allenai.org/scitail/): An entailment dataset created from multiple-choice science exams and web sentences.

**Named Entity Recognition (NER)**: labels sequences of words in a text which are the names of things, such as person and company names, or gene and protein names

- [CoNLL 2003 NER task](https://www.clips.uantwerpen.be/conll2003/): consists of newswire from the Reuters, concentrating on four types of named entities: persons, locations, organizations and names of miscellaneous entities.
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19): This corpus contains text in English, Arabic and Chinese, tagged with four different entity types (PER, LOC, ORG, MISC).
- [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html): A large collection of Reuters News stories.
- Fine-Grained NER (FGN)

**Sentiment Analysis**

- [SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank)
- [IMDb](http://ai.stanford.edu/~amaas/data/sentiment/): A large dataset of movie reviews with binary sentiment classification labels.

**Semantic Role Labeling (SRL)**: models the predicate-argument structure of a sentence, and is often described as answering “Who did what to whom”.

- [CoNLL-2004 & CoNLL-2005](http://www.lsi.upc.edu/~srlconll/)

**Sentence similarity**: also known as *paraphrase detection*

- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (MicRosoft Paraphrase Corpus): It contains pairs of sentences extracted from news sources on the web, with annotations indicating whether each pair is semantically equivalent.
- [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Quora Question Pairs) STS Benchmark: Semantic Textual Similarity

**Sentence Acceptability**: a task to annotate sentences for grammatical acceptability.

- [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability): a binary single-sentence classification task.

**Text Chunking**: To divide a text in syntactically correlated parts of words.

- [CoNLL-2000](https://www.clips.uantwerpen.be/conll2000/chunking/)

**Part-of-Speech (POS) Tagging**: tag parts of speech to each token, such as noun, verb, adjective, etc. the Wall Street Journal portion of the Penn Treebank (Marcus et al., 1993).

**Machine Translation**: See [Standard NLP](https://nlp.stanford.edu/projects/nmt/) page.

- WMT 2015 English-Czech data (Large)
- WMT 2014 English-German data (Medium)
- IWSLT 2015 English-Vietnamese data (Small)

**Coreference Resolution**: cluster mentions in text that refer to the same underlying real world entities.

- [CoNLL-2012](http://conll.cemantix.org/2012/data.html)

**Long-range Dependency**

- [LAMBADA](http://clic.cimec.unitn.it/lambada/) (LAnguage Modeling Broadened to Account for Discourse Aspects): A collection of narrative passages extracted from the BookCorpus and the task is to predict the last word, which require at least 50 tokens of context for a human to successfully predict.
- [Children’s Book Test](https://research.fb.com/downloads/babi/): is built from books that are freely available in [Project Gutenberg](https://www.gutenberg.org/). The task is to predict the missing word among 10 candidates.

**Multi-task benchmark**

- GLUE multi-task benchmark: [https://gluebenchmark.com](https://gluebenchmark.com/)
- decaNLP benmark: [https://decanlp.com](https://decanlp.com/)

**Unsupervised pretraining dataset**

- [Books corpus](https://googlebooks.byu.edu/): The corpus contains “over 7,000 unique unpublished books from a variety of genres including Adventure, Fantasy, and Romance.”
- [1B Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
- [English Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia): ~2500M words
