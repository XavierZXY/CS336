# CS336 Spring 2025 Assignment 1: Basics

## Train BPE Tokenizer
> [BPE tokenizer](cs336_basics/tokenizer.py)

### Train BPE

首先我们先对齐作业中定义好的接口函数，然后实现基本的 BPE 训练功能。
```python
def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_token: list[str], **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```

### Encode and Decode