# CS336 Spring 2025 Assignment 1: Basics

## Todo list
- [ ] Speed up the BPE algorithm.

## Train BPE Tokenizer
> [BPE tokenizer](cs336_basics/BPETokenizer.py)

### Train BPE

首先我们先对齐作业中定义好的接口函数，然后实现基本的 BPE 训练功能。
```python
def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_token: list[str], **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```
具体的流程如下：
1. 文件切分.
当有较长的文本时，可以将其切分为多个较短的片段，以便于多线程处理。
2. Seperating text into pretokens.
首先将输入文本进行子词切分，得到初始的子词单元。这里使用一般使用正则来切分字词单元.`PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
```
"some text that i'll pre-tokenize"
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```
3. 统计每个子词的频率。
    统计所有子词在训练数据中出现的频率，并记录下来。

4. merge the most frequent pairs.
根据子词的频率信息，合并出现频率最高的子词对，形成新的子词单元。这是最耗时的部分。目前还是参考朴素版本的实现，并不能通过`test_train_bpe_speed`，后续将通过大根堆等方式来改进。
合并过程之中，还要注意合并后的字词单元给其原来左右邻居频率带来的改变，以及特殊情况。

### Encode and Decode
通过 BPE 编码器对文本进行编码和解码。具体流程如下：
1. 文本编码
   使用训练好的 BPE 模型对输入文本进行编码，得到对应的子词序列。

2. 文本解码
   使用训练好的 BPE 模型对编码后的子词序列进行解码，恢复原始文本。

## Transformer Language Model Architecture

### Releative Positional Embeddings
Rope是这里最难理解的一个部分。我参考了以下文章来进行学习，先推一遍公式，再结合代码仔细推导，收获颇丰。
> [一文看懂 LLaMA 中的旋转式位置编码](https://zhuanlan.zhihu.com/p/642884818)
> [十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)

# Cite

```
https://github.com/Spectual/stanford-cs336-a1
```