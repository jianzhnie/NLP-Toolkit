# **Tokenizer**

1. 背景与基础

在使用GPT BERT模型输入词语常常会先进行tokenize ，tokenize具体目标与粒度是什么呢？tokenize也有许多类别及优缺点，这篇文章总结一下各个方法及实际案例。

tokenize的目标是把输入的文本流，切分成一个个子串，每个子串相对有完整的语义，便于学习embedding表达和后续模型的使用。

tokenize有三种粒度：**word/subword/char**

- **word词**，是最自然的语言单元。对于英文等自然语言来说，存在着天然的分隔符，比如说空格，或者是一些标点符号，对词的切分相对容易。但是对于一些东亚文字包括中文来说，就需要某种分词算法才行。顺便说一下，Tokenizers 库中，基于规则切分部分，采用了spaCy和Moses两个库。如果基于词来做词汇表，由于长尾现象的存在，这个词汇表可能会超大。像Transformer XL库就用到了一个26.7万个单词的词汇表。这需要极大的embedding matrix才能存得下。embedding matrix是用于查找取用token的embedding vector的。这对于内存或者显存都是极大的挑战。常规的词汇表，一般大小不超过5万。
- **char/字符,** 也就是说，我们的词汇表里只有最基本的字符。而一般来讲，字符的数量是少量有限的。这样做的问题是，由于字符数量太小，我们在为每个字符学习嵌入向量的时候，每个向量就容纳了太多的语义在内，学习起来非常困难。
- **subword子词级**，它介于字符和单词之间。比如说Transformers可能会被分成Transform和ers两个部分。**这个方案平衡了词汇量和语义独立性，是相对较优的方案**。它的处理原则是，**常用词应该保持原状，生僻词应该拆分成子词以共享token压缩空间**。



## Subword tokenization

Subword tokenization的核心思想是：“**频繁出现了词不应该被切分成更小的单位，但不常出现的词应该被切分成更小的单位**”。

比方"annoyingly"这种词，就不是很常见，但是"annoying"和"ly"都很常见，因此细分成这两个sub-word就更合理。中文也是类似的，比如“仓库管理系统”作为一个单位就明显在语料中不会很多，因此分成“仓库”和“管理系统”就会好很多。

这样分词的好处在于，大大节省了词表空间，还能够解决 OOV 问题。因为我们很多使用的词语，都是由更简单的词语或者词缀构成的，我们不用去保存那些“小词”各种排列组合形成的千变万化的“大词”，而用较少的词汇，去覆盖各种各样的词语表示。同时，相比与直接使用最基础的“字”作为词表，sub-word的语义表示能力也更强。

常见的 subword tokenization方法有：

- BPE
- WordPiece
- Unigram
- SentencePiece
- ...

| Model                                        | Type of Token  |
| -------------------------------------------- | -------------- |
| Bert                                         | WordPiece      |
| GPT, gpt-2                                   | Byte-level BPE |
| T5, ALBERT, CamemBERT, XLM-RoBERTa and XLNet | SentencePiece  |
| T5                                           | Unigram        |
|                                              |                |

这里对BPE做一个简单的介绍，让我们对 sub-word tokenization 的原理有一个基本了解：

## Byte-Pair Encoding (BPE)

BPE  — a frequency-based model

- Byte Pair Encoding uses the frequency of subword patterns to shortlist them for merging.
- The drawback of using frequency as the driving factor is that you can end up having ambiguous final encodings that might not be useful for the new input text.
- It still has the scope of improvement in terms of generating unambiguous tokens.

#### **Step1：首先，我们需要对语料进行一个预分词（pre-tokenization）：**

比方对于英文，我可以直接简单地使用空格加一些标点符号来分词；中文可以使用jieba或者直接字来进行分词。

分词之后，我们就得到了一个**原始词集合**，同时，统计每个词出现的频次供后续计算使用。

假设我们的词集合以及词频是：

```javascript
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

#### **Step2：构建基础词表（base vocab） 并开始学习 结合规则（merge rules）：**

对于英语来说，我们选择字母来构成**基础词表**：

```python
["b", "g", "h", "n", "p", "s", "u"]
```

注：这个基础词表，就是我们最终词表的初始状态，我们会不断构建新词，加进去，直到达到我们理想的词表规模。

接下来就是BPE的Byte-Pair核心部分 ——找symbol pair（符号对）并学习结合规则，根据规则，我们分别考察2-gram，3-gram的基本字符组合，把高频的ngram组合依次加入到词汇表当中，直到词汇表达到预定大小停止。

```javascript
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

即，我们从上面这个统计结果中，找出出现次数最多的那个符号对：

统计一下：

```javascript
h+u   出现了 10+5=15 次
u+g   出现了 10+5+5 = 20 次
p+u   出现了 12 次
...
```

统计完毕，我们发现`u+g`出现了最多次，因此，第一个结合规则就是：**把**`**u**`**跟**`**g**`**拼起来，得到**`**ug**`**这个新词！**

那么，我们就把`ug`加入到我们的基础词表：

```python
["b", "g", "h", "n", "p", "s", "u", "ug"]
```

同时，词频统计表也变成了：

```javascript
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

#### **Step3：反复地执行上一步，直到达到预设的词表规模。**

我们接着统计，发现下一个频率最高的 symbol pair是`u+n`，出现了12+4=16次，因此词表中增加`un`这个词；再下一个则是`h+ug`，出现了10+5=15次，因此添加`hug`这个词......

如此进行下去，当达到了预设的`vocab_size`的数目时，就停止，最终的词表就得到了！

最终词汇表的大小 = 基础字符词汇表大小 + 合并串的数量，比如像GPT，它的词汇表大小 40478 = 478(基础字符) + 40000（merges）。添加完后，我们词汇表变成：

```python
["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
```

实际使用中，如果遇到未知字符用<unk>代表

## Byte-level BPE

BPE的一个问题是，如果遇到了unicode，基本字符集可能会很大。一种处理方法是我们以一个字节为一种“字符”，不管实际字符集用了几个字节来表示一个字符。这样的话，基础字符集的大小就锁定在了256。

例如，像GPT-2的词汇表大小为50257 = 256 + <EOS> + 50000 mergers，<EOS>是句子结尾的特殊标记。

## WordPiece

- With the release of BERT in 2018, there came a new subword tokenization algorithm called WordPiece which can be considered as an intermediary of BPE and Unigram algorithms.
- WordPiece is also a greedy algorithm that leverages likelihood instead of count frequency to merge the best pair in each iteration but the choice of characters to pair is based on count frequency.
- So, it is similar to BPE in terms of choosing characters to pair and similar to Unigram in terms of choosing the best pair to merge.

WordPiece，从名字好理解，它是一种子词粒度的tokenize算法subword tokenization algorithm，很多著名的Transformers 模型，比如BERT/DistilBERT/Electra都使用了它。

它的原理非常接近BPE，不同之处在于，它在做合并的时候，并不是每次找最高频的组合，而是找能够**最大化训练集数据似然**的 merge，即它每次合并的两个字符串A和B，应该具有最大的 P(AB)/P(A)P(B)  值。合并AB之后，所有原来切成 A+B 两个tokens的就只保留AB一个token，整个训练集上最大似然变化量与 P(AB)/P(A)P(B)  成正比。

## Unigram

Unigram — a probability-based model

- Comes in the Unigram model that approaches to solve the merging problem by calculating the likelihood of each subword combination rather than picking the most frequent pattern.
- It calculates the probability of every subword token and then drops it based on a loss function that is explained in [this research paper.](https://arxiv.org/pdf/1804.10959.pdf)
- Based on a certain threshold of the loss value, you can then trigger the model to drop the bottom 20–30% of the subword tokens.
- Unigram is a completely probabilistic algorithm that chooses both the pairs of characters and the final decision to merge(or not) in each iteration based on probability.

与BPE或者WordPiece不同，Unigram的算法思想是从一个巨大的词汇表出发，再逐渐删除trim down其中的词汇，直到size满足预定义。

初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。

每次从词汇表中删除词汇的原则是使预定义的损失最小。训练时，计算loss的公式为：

Loss= ∑ log(∑p(x))

 假设训练文档中的所有词分别为 x1;x2...xN  ，而每个词tokenize的方法是一个集合 S(xi)  。当一个词汇表确定时，每个词tokenize的方法集合 S(xi) 就是确定的，而每种方法对应着一个概率p(x)。如果从词汇表中删除部分词，则某些词的tokenize的种类集合就会变少，log(*)中的求和项就会减少，从而增加整体loss。

Unigram算法每次会从词汇表中挑出使得loss增长最小的10%~20%的词汇来删除。

一般Unigram算法会与SentencePiece算法连用。

## SentencePiece

SentencePiece，顾名思义，它是把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格space也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。

比如，XLNetTokenizer就采用了_ 来代替空格，解码的时候会再用空格替换回来。

目前，Tokenizers库中，所有使用了SentencePiece的都是与Unigram算法联合使用的，比如ALBERT、XLNet、Marian和T5.



## 切分实例与代码分析

下面，我们就直接使用Tokenizer来进行分词：

```javascript
from transformers import BertTokenizer  # 或者 AutoTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
s = 'today is a good day to learn transformers'
tokenizer()
```

得到：

```javascript
{'input_ids': [101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

### **了解一下内部的具体步骤：**

1. `tokenize()`

```javascript
s = 'today is a good day to learn transformers'
tokens = tokenizer.tokenize(s)
tokens
```

输出：

```javascript
['today', 'is', 'a', 'good', 'day', 'to', 'learn', 'transform', '##ers']
```

注意这里的分词结果，`transformers`被分成了`transform`和`##ers`。这里的##代表这个词应该紧跟在前面的那个词，组成一个完整的词。

这样设计，主要是为了方面我们在还原句子的时候，可以正确得把sub-word组成成原来的词。

1. `convert_tokens_to_ids()`

```javascript
ids = tokenizer.convert_tokens_to_ids(tokens)
ids
```

输出：

```javascript
[2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468]
```

1. `decode`

```javascript
print(tokenizer.decode([1468]))
print(tokenizer.decode(ids))  # 注意这里会把subword自动拼起来
```

输出：

```javascript
##ers
today is a good day to learn transformers
```

### **Special Tokens**

观察一下上面的结果，直接call tokenizer得到的ids是：

```javascript
[101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102]
```

而通过`convert_tokens_to_ids`得到的ids是：

```javascript
[2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468]
```

可以发现，前者在头和尾多了俩token，id分别是 101 和 102。

decode出来瞅瞅：

```javascript
tokenizer.decode([101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102])
```

输出：

```javascript
'[CLS] today is a good day to learn transformers [SEP]'
```

它们分别是 `[CLS]` 和 `[SEP]`。这两个token的出现，是因为我们调用的模型，在pre-train阶段使用了它们，所以tokenizer也会使用。

不同的模型使用的special tokens不一定相同，所以一定要让tokenizer跟model保持一致！

