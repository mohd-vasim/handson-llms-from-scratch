# Mastering the `dim` Parameter in PyTorch

> A comprehensive, intuitive, and practical guide to never getting confused by PyTorch dimensions again — with deep dives into **Chapter 4** of *Build a Large Language Model (From Scratch)* (`ch04.ipynb`) and modern LLM architectures.

---

## Table of Contents
1. [The Core Mental Model: The Golden Rule of `dim`](#1-the-core-mental-model-the-golden-rule-of-dim)
2. [Tensor Coordinates & Negative Indexing](#2-tensor-coordinates--negative-indexing)
3. [Operation Category 1: Reductions (`mean`, `var`, `sum`, `argmax`)](#3-operation-category-1-reductions-mean-var-sum-argmax)
   - [The Shape Collapse Rule](#the-shape-collapse-rule)
   - [Why `keepdim=True` is Essential](#why-keepdimtrue-is-essential)
4. [Operation Category 2: Transformations (`softmax`, `log_softmax`)](#4-operation-category-2-transformations-softmax-log_softmax)
5. [Operation Category 3: Joining Tensors (`torch.cat` vs `torch.stack`)](#5-operation-category-3-joining-tensors-torchcat-vs-torchstack)
6. [Operation Category 4: Reshaping & Slicing (`squeeze`, `unsqueeze`, `split`, `chunk`)](#6-operation-category-4-reshaping--slicing-squeeze-unsqueeze-split-chunk)
7. [Deep Dive: Every `dim` Usage in `ch04.ipynb`](#7-deep-dive-every-dim-usage-in-ch04ipynb)
   - [Case A: `torch.stack(batch, dim=0)`](#case-a-torchstackbatch-dim0)
   - [Case B: LayerNorm `out.mean(dim=-1, keepdim=True)`](#case-b-layernorm-outmeandim-1-keepdimtrue)
   - [Case C: Sampling `torch.softmax(logits, dim=-1)`](#case-c-sampling-torchsoftmaxlogits-dim-1)
   - [Case D: Prediction `torch.argmax(probas, dim=-1, keepdim=True)`](#case-d-prediction-torchargmaxprobas-dim-1-keepdimtrue)
   - [Case E: Autoregressive Append `torch.cat((idx, idx_next), dim=1)`](#case-e-autoregressive-append-torchcatidx-idx_next-dim1)
   - [Case F: Squeezing Batch Dim `out.squeeze(0)`](#case-f-squeezing-batch-dim-outsqueeze0)
8. [Other Essential LLM Usages (Attention, Cross-Entropy, Projections)](#8-other-essential-llm-usages-attention-cross-entropy-projections)
9. [Quick-Reference Cheat Sheet & Decision Tree](#9-quick-reference-cheat-sheet--decision-tree)

---

## 1. The Core Mental Model: The Golden Rule of `dim`

The most common reason people get confused by `dim` is the false intuition:
> *"If `dim=0` represents rows, does `x.sum(dim=0)` calculate the sum of each row?"*  
> **NO! It does the exact opposite.**

### The Golden Rule
> **The `dim` you specify is the axis that gets COLLAPSED (in reductions) or WALKED ALONG (in softmax/argmax) or EXPANDED / JOINED (in stack/cat).**

Think of it like squishing an accordion:
- If you specify `dim=0`, you squish the tensor along axis 0 (downwards across rows). The rows collapse into one row; what remains are the columns.
- If you specify `dim=1`, you squish the tensor along axis 1 (horizontally across columns). The columns collapse into one column; what remains are the rows.

```
2D Tensor (Shape: [2, 3]):
             dim=1 (Columns) --->
             Col 0   Col 1   Col 2
dim=0   Row 0 [ 1.0,   2.0,   3.0 ]
  |     Row 1 [ 4.0,   5.0,   6.0 ]
  v
(Rows)

- x.sum(dim=0): Collapse Row 0 & Row 1  ==> [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0] (Shape: [3])
- x.sum(dim=1): Collapse Col 0, 1 & 2   ==> [1+2+3, 4+5+6] = [6.0, 15.0]     (Shape: [2])
```

---

## 2. Tensor Coordinates & Negative Indexing

In deep learning and NLP, tensors frequently have 3 or 4 dimensions:

```
3D LLM Tensor: [batch_size, sequence_length, embedding_dim]
Positive dim:        0               1               2
Negative dim:       -3              -2              -1
```

```
4D Attention Tensor: [batch_size, num_heads, sequence_length, head_dim]
Positive dim:              0           1            2             3
Negative dim:             -4          -3           -2            -1
```

### Why Negative Indexing (`dim=-1`, `dim=-2`) is Preferred in LLMs
- **`dim=-1` always means the feature/embedding/vocabulary dimension**, regardless of whether your tensor is 2D `[batch, emb_dim]`, 3D `[batch, seq_len, emb_dim]`, or 4D `[batch, heads, seq_len, head_dim]`.
- **`dim=-2` always means the sequence length (tokens) dimension** in 3D and 4D representations.
- **`dim=0` almost always means the batch dimension**.

By using `dim=-1`, your code is robust and does not break if a batch or head dimension is added or removed earlier in the pipeline.

---

## 3. Operation Category 1: Reductions (`mean`, `var`, `sum`, `argmax`)

### The Shape Collapse Rule
When you perform a reduction (`.sum(dim=k)`, `.mean(dim=k)`, `.max(dim=k)`), **dimension `k` disappears from the output shape**:

$$\text{Input shape: } (d_0, d_1, \dots, d_k, \dots, d_{n-1})$$
$$\text{Output shape (keepdim=False): } (d_0, d_1, \dots, d_{k-1}, d_{k+1}, \dots, d_{n-1})$$

#### Example:
```python
import torch

x = torch.randn(2, 4, 768)  # [batch_size=2, seq_len=4, emb_dim=768]

# Reduce along dim=0 (collapse batch)
out0 = x.mean(dim=0)
print(out0.shape)  # torch.Size([4, 768])

# Reduce along dim=1 (collapse sequence length: mean pooling across tokens)
out1 = x.mean(dim=1)
print(out1.shape)  # torch.Size([2, 768])

# Reduce along dim=-1 (collapse embedding dimension)
out2 = x.mean(dim=-1)
print(out2.shape)  # torch.Size([2, 4])
```

### Why `keepdim=True` is Essential

When `keepdim=False` (the default), PyTorch squeezes out the reduced dimension. When `keepdim=True`, the reduced dimension remains as size `1`.

```python
x = torch.randn(2, 5)  # 2 samples, 5 features

mean_no_keep = x.mean(dim=-1)
print(mean_no_keep.shape)  # torch.Size([2])

mean_keep = x.mean(dim=-1, keepdim=True)
print(mean_keep.shape)     # torch.Size([2, 1])
```

#### Why does this matter? **Broadcasting!**
Suppose you want to normalize `x` by subtracting its mean:
```python
# With keepdim=False:
diff = x - mean_no_keep
# x is [2, 5], mean_no_keep is [2]
# In PyTorch, broadcasting aligns trailing dimensions!
# PyTorch attempts to match dimension 5 with dimension 2 -> MISMATCH / CRASH!
# RuntimeError: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 1

# With keepdim=True:
diff = x - mean_keep
# x is [2, 5], mean_keep is [2, 1]
# [2, 1] automatically broadcasts across all 5 columns!
# Correct result: each row has its own mean subtracted.
```

---

## 4. Operation Category 2: Transformations (`softmax`, `log_softmax`)

Unlike reductions, transformations **preserve the original tensor shape**. The `dim` parameter here tells PyTorch:
> *"Along which dimension should the values normalize to sum to 1.0?"*

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [1.0, 1.0, 1.0]
])  # Shape: [2, 3]
```

### Comparison: `dim=-1` (across columns) vs `dim=0` (across rows)

```python
# 1. dim=-1 (or dim=1): Each ROW sums to 1.0
s_col = torch.softmax(x, dim=-1)
print(s_col)
# tensor([[0.0900, 0.2447, 0.6652],   <-- Sums to 1.0
#         [0.3333, 0.3333, 0.3333]])  <-- Sums to 1.0
print(s_col.sum(dim=-1))
# tensor([1.0000, 1.0000])

# 2. dim=0: Each COLUMN sums to 1.0 (competing across batch items)
s_row = torch.softmax(x, dim=0)
print(s_row)
# tensor([[0.5000, 0.7311, 0.8808],
#         [0.5000, 0.2689, 0.1192]])
#           |       |       |
#         (1.0)   (1.0)   (1.0)
```

In LLMs, you almost **never** want `dim=0` for softmax, because you want the probability distribution over tokens or vocabulary for *each* individual example independently.

---

## 5. Operation Category 3: Joining Tensors (`torch.cat` vs `torch.stack`)

This is one of the most frequent points of confusion. Both take a sequence of tensors and join them along a `dim`, but they differ fundamentally in dimension count.

| Operation | Action | Output Dimensions | Requirement |
| :--- | :--- | :--- | :--- |
| `torch.cat(tensors, dim=d)` | Glues tensors together along an **existing** dimension `d` | Same number of dimensions | Shapes must match everywhere **except** at `dim=d` |
| `torch.stack(tensors, dim=d)` | Inserts a **brand-new** dimension at index `d` and stacks along it | $\text{Number of dims} + 1$ | All input tensors must have the **exact same shape** |

### Visual Comparison

Suppose we have two 1D tensors of length 3:
```python
t1 = torch.tensor([1, 2, 3])  # Shape: [3]
t2 = torch.tensor([4, 5, 6])  # Shape: [3]

# torch.cat along existing dim 0:
cat_out = torch.cat([t1, t2], dim=0)
print(cat_out)        # tensor([1, 2, 3, 4, 5, 6])
print(cat_out.shape)  # torch.Size([6]) -> Still 1D!

# torch.stack along new dim 0:
stack_out0 = torch.stack([t1, t2], dim=0)
print(stack_out0)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(stack_out0.shape)  # torch.Size([2, 3]) -> Now 2D!

# torch.stack along new dim 1:
stack_out1 = torch.stack([t1, t2], dim=1)
print(stack_out1)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
print(stack_out1.shape)  # torch.Size([3, 2]) -> Now 2D!
```

---

## 6. Operation Category 4: Reshaping & Slicing (`squeeze`, `unsqueeze`, `split`, `chunk`)

### `torch.unsqueeze(x, dim)`
Adds a dimension of size `1` at index `dim`.
```python
x = torch.zeros(4, 768)  # [4, 768] (e.g. sequence of 4 token embeddings)

# Add a batch dimension at index 0:
x_batched = x.unsqueeze(dim=0)
print(x_batched.shape)   # [1, 4, 768]

# Add a dimension at the end:
x_trailing = x.unsqueeze(dim=-1)
print(x_trailing.shape)  # [4, 768, 1]
```

### `torch.squeeze(x, dim)`
Removes a dimension of size `1` at index `dim`. If the dimension is not of size 1, `squeeze` does nothing.
```python
x = torch.zeros(1, 4, 768)

out = x.squeeze(dim=0)
print(out.shape)         # [4, 768] (Batch dimension removed)
```

### `torch.chunk(x, chunks, dim)`
Splits tensor `x` into `chunks` pieces along dimension `dim`.
```python
# Common in multi-head attention: projecting Q, K, V in one large linear layer
qkv = torch.randn(2, 4, 3 * 768)  # [2, 4, 2304]
q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
print(q.shape, k.shape, v.shape)  # All [2, 4, 768]
```

---

## 7. Deep Dive: Every `dim` Usage in `ch04.ipynb`

In `LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb`, the `dim` parameter appears in six critical locations. Let's analyze each one in depth.

```
                               ch04.ipynb Execution Flow
                               -------------------------
+----------------------------------------------------------------------------------+
| 1. DataLoader batching:     torch.stack(batch, dim=0)                            |
|                             List of [context_len]  ==>  [batch_size, context_len]|
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| 2. LayerNorm:               out.mean(dim=-1, keepdim=True)                       |
|                             Normalizes across feature dimension [emb_dim]        |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| 3. Next-Token Probabilities:torch.softmax(logits, dim=-1)                        |
|                             Probabilities sum to 1.0 across [vocab_size]         |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| 4. Greedy Selection:        torch.argmax(probas, dim=-1, keepdim=True)           |
|                             Picks winning token ID; preserves shape [batch, 1]   |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| 5. Autoregressive Append:   torch.cat((idx, idx_next), dim=1)                    |
|                             Glues [batch, seq_len] + [batch, 1] along seq axis   |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| 6. Final Detokenization:    out.squeeze(0)                                       |
|                             Strips batch dim: [1, seq_len]  ==>  [seq_len]       |
+----------------------------------------------------------------------------------+
```

---

### Case A: `torch.stack(batch, dim=0)`
*(Found around line 254 in notebook)*

```python
# Inside the dataset/dataloader preparation:
batch = [tensor_sample_1, tensor_sample_2, tensor_sample_3]
# Each tensor_sample has shape: [context_length] (1D tensor of token IDs)

batch = torch.stack(batch, dim=0)
# Resulting shape: [batch_size, context_length]
```

#### Why `dim=0`?
We want each sample in the list to become an entry along the **new batch axis** (index 0).
- If we had called `torch.cat(batch, dim=0)`, the samples would have merged end-to-end into one long 1D array of length `batch_size * context_length`.
- `torch.stack(batch, dim=0)` creates a 2D batch matrix: row 0 is sample 1, row 1 is sample 2, etc.

---

### Case B: LayerNorm `out.mean(dim=-1, keepdim=True)`
*(Found in lines 412–413, 533–534, 591–592)*

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

#### Why `dim=-1`?
In Transformer architectures, tensor `x` has shape `[batch_size, num_tokens, emb_dim]`.
- **Layer Normalization** requires normalizing across the **features** of each token independently.
- Each token vector of size `emb_dim` should independently have zero mean and unit variance.
- Because `emb_dim` is the last axis, `dim=-1` isolates the feature values for that single token.

#### Why NOT `dim=0`?
`dim=0` is what **Batch Normalization** uses! BatchNorm computes the mean and variance across the batch. In NLP with varying sequence lengths and autoregressive inference (batch size = 1), BatchNorm breaks down. LayerNorm computes statistics across `dim=-1`, completely independent of other tokens and other batch items.

#### Why `keepdim=True`?
- Shape of `x`: `[batch_size, num_tokens, emb_dim]`
- Without `keepdim=True`, `mean` has shape `[batch_size, num_tokens]`.
- Subtracting `[batch_size, num_tokens]` from `[batch_size, num_tokens, emb_dim]` fails or requires manual reshaping.
- With `keepdim=True`, `mean` has shape `[batch_size, num_tokens, 1]`.
- Broadcasting automatically expands `[batch_size, num_tokens, 1]` across all `emb_dim` channels seamlessly.

---

### Case C: Sampling `torch.softmax(logits, dim=-1)`
*(Found in line 1413 in the `generate_text_simple` function)*

```python
# Focus only on the last token: logits has shape [batch_size, vocab_size]
logits = logits[:, -1, :]  

# Apply softmax to get probabilities
probas = torch.softmax(logits, dim=-1)  # shape: [batch_size, vocab_size]
```

#### Why `dim=-1`?
The vocabulary size `vocab_size` (e.g., 50,257 in GPT-2) is the last dimension.
- We want each row (each sequence in the batch) to form a valid categorical probability distribution:
  $$\sum_{v=0}^{\text{vocab\_size}-1} \text{probas}[b, v] = 1.0$$
- `dim=-1` instructs softmax to exponentiate every logit across the vocabulary axis and divide by the sum along that same axis.

---

### Case D: Prediction `torch.argmax(probas, dim=-1, keepdim=True)`
*(Found in line 1416)*

```python
# Get the idx of the vocab entry with the highest probability value
idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # shape: [batch_size, 1]
```

#### Why `dim=-1`?
`argmax` scans along the specified axis and returns the integer index of the maximum value.
- Scanning along `dim=-1` asks: *"Which word index in the vocabulary had the highest probability for this batch item?"*

#### Why `keepdim=True`?
- `probas` has shape `[batch_size, vocab_size]`.
- With `keepdim=False`: `idx_next` would have shape `[batch_size]`.
- With `keepdim=True`: `idx_next` has shape `[batch_size, 1]`.
- Having shape `[batch_size, 1]` is immediately ready for concatenation with the historical tokens matrix `idx` of shape `[batch_size, current_seq_len]`.

---

### Case E: Autoregressive Append `torch.cat((idx, idx_next), dim=1)`
*(Found in line 1419)*

```python
# Append sampled index to the running sequence
idx = torch.cat((idx, idx_next), dim=1)  # shape: [batch, n_tokens + 1]
```

#### Why `dim=1`?
Let's inspect the shapes:
- `idx` shape: `[batch_size, n_tokens]`
- `idx_next` shape: `[batch_size, 1]`

We are generating text token-by-token along the **sequence length axis**, which is dimension 1:
- Dimension 0: `batch_size` (must match!)
- Dimension 1: `n_tokens` (getting extended: $n + 1$)

```
idx:
[ [token_0, token_1, ..., token_n],
  [token_0, token_1, ..., token_n] ]  <-- Shape: [2, n]

idx_next:
[ [new_token],
  [new_token] ]                       <-- Shape: [2, 1]

torch.cat((idx, idx_next), dim=1):
[ [token_0, token_1, ..., token_n, new_token],
  [token_0, token_1, ..., token_n, new_token] ]  <-- Shape: [2, n + 1]
```

If we accidentally set `dim=0`, PyTorch would try to concatenate along the batch axis, which would fail with a shape mismatch because $n\_tokens \neq 1$.

---

### Case F: Squeezing Batch Dim `out.squeeze(0)`
*(Found in line 1519)*

```python
# out has shape [1, total_generated_tokens]
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
```

#### Why `squeeze(0)`?
The model generates outputs with a batch dimension: `[1, seq_len]`.
The tokenizer's `.decode(...)` method expects a flat 1D Python list of token integers `[token_1, token_2, ...]`.
- `out.squeeze(0)` explicitly eliminates dimension 0 (the single batch item), resulting in shape `[total_generated_tokens]`.
- Calling `out.squeeze(0)` is safer than calling `out.squeeze()` with no arguments, because `out.squeeze()` would also collapse the sequence dimension if `total_generated_tokens == 1`.

---

## 8. Other Essential LLM Usages (Attention, Cross-Entropy, Projections)

### A. Scaled Dot-Product Attention (Chapter 3)
```python
# attention_scores shape: [batch_size, num_heads, num_tokens, num_tokens]
# For each query token (row), compute attention weights across key tokens (columns):
attention_weights = torch.softmax(attention_scores, dim=-1)
```
- The last dimension represents keys ($K^T$).
- Each query token assigns a distribution summing to 1.0 across all key tokens.

### B. Loss Computation (`F.cross_entropy`)
`nn.CrossEntropyLoss` expects logits of shape:
- Either `[N, C]` for 1D targets `[N]`.
- Or `[N, C, d1, d2]` where `C` is the class/vocab dimension.

In LLMs:
```python
# logits shape: [batch_size, seq_len, vocab_size]
# targets shape: [batch_size, seq_len]

# Flatten batch and sequence dimensions before loss:
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
# Now logits is [batch_size * seq_len, vocab_size] -> class dim is dim=-1 (dim 1)
```

### C. Finding Top-K Tokens
```python
# topk along vocab dimension:
top_values, top_indices = torch.topk(logits, k=5, dim=-1)
```
- Retrieves the 5 highest logit values and their corresponding token IDs for each token position.

---

## 9. Quick-Reference Cheat Sheet & Decision Tree

### The 3-Second Mental Checklist
Whenever you write `dim=...`, ask yourself:

1. **What is my tensor's shape?**  
   *Write it down:* `[batch, seq_len, emb_dim]`.
2. **Which axis do I want to eliminate, normalize, or extend?**  
   - Want to combine tokens into one sentence representation? Collapse `seq_len` $\to$ `dim=1` (or `dim=-2`).
   - Want to normalize each token's vector? Collapse/normalize `emb_dim` $\to$ `dim=-1`.
   - Want to append the next token? Extend `seq_len` $\to$ `dim=1`.
   - Want to combine examples into a batch? New dimension $\to$ `torch.stack(..., dim=0)`.
3. **Do I need to broadcast afterward?**  
   - If YES $\to$ always add `keepdim=True`.

### Quick Operation Lookup Table

| Goal | Method | `dim` Value | Example Shape Change |
| :--- | :--- | :--- | :--- |
| Normalize across embedding (LayerNorm) | `.mean()`, `.var()` | `dim=-1, keepdim=True` | `[2, 4, 768] -> [2, 4, 1]` |
| Pool sequence into single vector | `.mean()` | `dim=1` | `[2, 4, 768] -> [2, 768]` |
| Convert logits to probabilities | `torch.softmax()` | `dim=-1` | `[2, 50257] -> [2, 50257]` |
| Pick most likely next token | `torch.argmax()` | `dim=-1, keepdim=True` | `[2, 50257] -> [2, 1]` |
| Append new token to sequence | `torch.cat()` | `dim=1` | `([2, 10], [2, 1]) -> [2, 11]` |
| Stack samples into a batch | `torch.stack()` | `dim=0` | `3 x [10] -> [3, 10]` |
| Add batch dimension | `.unsqueeze()` | `dim=0` | `[10] -> [1, 10]` |
| Remove batch dimension | `.squeeze()` | `dim=0` | `[1, 10] -> [10]` |
| Multi-Head Attention softmax | `torch.softmax()` | `dim=-1` | `[B, H, S, S] -> [B, H, S, S]` |
| Split Q, K, V projections | `torch.chunk()` | `dim=-1` | `[B, S, 3*D] -> 3 x [B, S, D]` |

---

*Keep this guide handy as you build and train models from scratch!*

