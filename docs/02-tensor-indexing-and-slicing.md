# Demystifying PyTorch Tensor Slicing & Indexing: `[:, -1, :]` and Beyond

> A definitive, step-by-step guide to mastering tensor indexing patterns in PyTorch, with detailed breakdowns of `[:, -1, :]`, `[:, -context_size:]`, `[:, :-1]`, ellipsis (`...`), and new axis injection (`None`).

---

## Table of Contents
1. [The Foundational Rule: Commas Separate Dimensions](#1-the-foundational-rule-commas-separate-dimensions)
2. [Anatomy of `logits[:, -1, :]`](#2-anatomy-of-logits--1-)
   - [What each position means](#what-each-position-means)
   - [Visual diagram](#visual-diagram)
   - [Why only the last token position `[-1]` is kept](#why-only-the-last-token-position--1-is-kept)
3. [The #1 Trap: Integer Indexing vs. Slice Indexing (`-1` vs `-1:`)](#3-the-1-trap-integer-indexing-vs-slice-indexing--1-vs--1)
   - [Dropping a dimension vs. Keeping a dimension](#dropping-a-dimension-vs-keeping-a-dimension)
   - [When does this bite you?](#when-does-this-bite-you)
4. [Mastering the Colon (`:`) in Deep Learning](#4-mastering-the-colon--in-deep-learning)
   - [Bare colon `:` (Take everything)](#bare-colon--take-everything)
   - [Sliding context window: `idx[:, -context_size:]`](#sliding-context-window-idx--context_size)
   - [The Next-Token Shift: `x[:, :-1]` and `targets[:, 1:]`](#the-next-token-shift-x--1-and-targets-1)
5. [The Ellipsis (`...`) — Handling Arbitrary Dimensions](#5-the-ellipsis----handling-arbitrary-dimensions)
6. [Inserting Dimensions On The Fly: `None` / `np.newaxis`](#6-inserting-dimensions-on-the-fly-none--npnewaxis)
7. [Step-by-Step Walkthrough: The Generation Loop in `ch04.ipynb`](#7-step-by-step-walkthrough-the-generation-loop-in-ch04ipynb)
8. [Popular Slicing Patterns Across LLMs (BERT, GPT, Attention)](#8-popular-slicing-patterns-across-llms-bert-gpt-attention)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. The Foundational Rule: Commas Separate Dimensions

In standard Python lists, multi-dimensional indexing requires multiple chained brackets:
```python
# Python 2D list:
val = my_list[i][j]
```

In PyTorch (and NumPy), **dimensions are separated by commas inside a single set of brackets**:
```python
# PyTorch 3D tensor:
val = my_tensor[dim_0, dim_1, dim_2]
```

When you see brackets with commas:
```
tensor[   :   ,   -1   ,   :   ]
          |        |       |
       Axis 0    Axis 1  Axis 2
       (Batch)   (Time)  (Features)
```
Each comma moves you to the next dimension from outermost to innermost:
- **Axis 0 (1st slot):** Which batch item(s)?
- **Axis 1 (2nd slot):** Which token/time step(s)?
- **Axis 2 (3rd slot):** Which feature/logit(s)?

---

## 2. Anatomy of `logits[:, -1, :]`

In `ch04.ipynb` (line 1410), during autoregressive text generation:
```python
# Focus only on the last time step
# (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
logits = logits[:, -1, :]
```

### What Each Position Means

| Slot | Syntax | Meaning | Effect on Shape |
| :---: | :---: | :--- | :--- |
| **0** | `:` | **Full slice across Batch**: Keep all batch items | Batch dimension is **preserved** |
| **1** | `-1` | **Single index across Sequence**: Select only the very last token position | Sequence dimension is **collapsed (dropped)** |
| **2** | `:` | **Full slice across Vocab**: Keep all vocabulary logit scores | Vocab dimension is **preserved** |

```
Input Shape:   [batch_size, n_tokens, vocab_size]  (e.g., [2, 4, 50257])
                    |           |          |
Index applied:     [:]        [-1]        [:]
                    |           |          |
                    v           v          v
Output Shape:  [batch_size,          vocab_size]   (e.g., [2, 50257])
```

### Visual Diagram

Imagine a batch of 2 sequences, each with 4 tokens, and a small vocabulary of 5 words:

```
Batch 0:
Token 0: [ 1.2, -0.5,  0.1,  2.3, -1.0 ]
Token 1: [ 0.1,  0.4, -0.2,  1.1,  0.8 ]
Token 2: [ 2.0,  1.5, -0.8, -0.1,  0.3 ]
Token 3: [ 0.5, -1.2,  3.4,  0.9, -0.2 ]  <-- Last token (-1)

Batch 1:
Token 0: [ 0.2,  0.1,  0.9, -0.4,  1.1 ]
Token 1: [ 1.8, -0.3,  0.5,  0.2, -1.4 ]
Token 2: [ -0.1, 2.2,  0.0,  1.0,  0.7 ]
Token 3: [ 3.1,  0.2, -1.5,  2.0,  0.4 ]  <-- Last token (-1)
```

Applying `logits[:, -1, :]`:
1. The `:` in dimension 0 grabs both Batch 0 and Batch 1.
2. The `-1` in dimension 1 ignores Token 0, Token 1, and Token 2, isolating **only Token 3**.
3. The `:` in dimension 2 retains all 5 logit scores for that token.

The result is a clean 2D matrix:
```
tensor([[0.5, -1.2,  3.4,  0.9, -0.2],   # Batch 0 last token logits
        [3.1,  0.2, -1.5,  2.0,  0.4]])  # Batch 1 last token logits
Shape: [2, 5]
```

### Why Only the Last Token Position (`-1`) is Kept

In a causal decoder-only LLM (like GPT):
- Every token position $t$ produces predictions for token $t+1$.
- Token position 0 predicts Token 1.
- Token position 1 predicts Token 2.
- ...
- **The final token position $N-1$ (`-1`) predicts the brand new next token $N$!**

All previous token predictions have already occurred in past steps. When generating new text, the only prediction that matters right now is the one emitted from the last token.

---

## 3. The #1 Trap: Integer Indexing vs. Slice Indexing (`-1` vs `-1:`)

This is the single most common cause of tensor shape bugs in PyTorch.

```python
x = torch.randn(2, 4, 768)  # [batch, tokens, emb]
```

Compare these two operations:

```python
# 1. Integer Indexing (-1)
out_int = x[:, -1, :]
print(out_int.shape)
# Output: torch.Size([2, 768])  <-- 3D became 2D! (Dimension collapsed)

# 2. Slice Indexing (-1:)
out_slice = x[:, -1:, :]
print(out_slice.shape)
# Output: torch.Size([2, 1, 768]) <-- Still 3D! (Dimension kept as size 1)
```

### The Golden Rule of Indexing vs Slicing
- **Integer index (`k` or `-1`)**: "Extract the single element at index $k$ and **eliminate** that dimension."
- **Slice (`k:k+1` or `-1:`)**: "Extract a range of elements and **keep** the dimension."

### When Does This Bite You?

Suppose a downstream module expects a 3D sequence tensor `[batch, seq_len, emb_dim]`:
```python
# If you used integer indexing:
single_token = x[:, -1, :]  # Shape: [2, 768] (2D)
# Passing single_token into self_attention(single_token) will crash:
# Expected 3D tensor [batch, seq, emb], got 2D tensor!

# If you used slice indexing:
single_token = x[:, -1:, :] # Shape: [2, 1, 768] (3D with seq_len=1)
# self_attention(single_token) works perfectly!
```

---

## 4. Mastering the Colon (`:`) in Deep Learning

The colon notation `start:stop:step` provides powerful ways to window, crop, and shift tensors.

### Bare Colon `:` (Take Everything)
- `x[:]` takes everything along that dimension (equivalent to `slice(None)`).
- If a tensor is 3D, writing `x[:, -1, :]` explicitly states: "all of dim 0, the last of dim 1, all of dim 2".
- Note: in PyTorch, trailing bare colons can be omitted: `x[:, -1]` is equivalent to `x[:, -1, :]`. However, writing `x[:, -1, :]` is preferred in production code for visual clarity!

---

### Sliding Context Window: `idx[:, -context_size:]`
*(Found in `ch04.ipynb` line 1402)*

```python
# E.g., if LLM supports only 5 tokens, and the current sequence has 10 tokens:
idx_cond = idx[:, -context_size:]
```

#### How Negative Slice Indexing Works:
- In Python slicing, `-k:` means **"start from $k$ elements before the end, and go all the way to the end"**.
- Example: If `context_size = 3` and sequence has token IDs `[10, 20, 30, 40, 50]`:
  - `idx[:, -3:]` extracts tokens `[30, 40, 50]`.
- **Batch axis (`:`)**: Keeps all sequences in the batch intact while windowing their tokens simultaneously.

```
idx (Shape: [2, 5]):
[ [101, 102, 103, 104, 105],
  [201, 202, 203, 204, 205] ]

idx[:, -3:] (Shape: [2, 3]):
[ [103, 104, 105],
  [203, 204, 205] ]
```

---

### The Next-Token Shift: `x[:, :-1]` and `targets[:, 1:]`
*(Standard causal LM training pattern)*

When training a language model, the inputs and targets are created from the same text sequence by shifting one token position to the right:

```python
# Suppose sequence has 5 tokens: ["The", "cat", "sat", "on", "mat"]
tokens = torch.tensor([[10, 20, 30, 40, 50]])  # [1, 5]

# Input: All tokens EXCEPT the last one (0 up to length - 1)
inputs  = tokens[:, :-1]  # tensor([[10, 20, 30, 40]]) -> ["The", "cat", "sat", "on"]

# Target: All tokens EXCEPT the first one (from index 1 to end)
targets = tokens[:, 1:]   # tensor([[20, 30, 40, 50]]) -> ["cat", "sat", "on", "mat"]
```

```
Tokens:    [ 10,  20,  30,  40,  50 ]
              \    \    \    \
Inputs:    [ 10,  20,  30,  40 ]      <-- tokens[:, :-1]
Targets:   [ 20,  30,  40,  50 ]      <-- tokens[:, 1:]
```
For every input token $x_t$, the model must predict target $y_t = x_{t+1}$.

---

## 5. The Ellipsis (`...`) — Handling Arbitrary Dimensions

The ellipsis (`...`) means: **"expand into as many `:` (full slices) as necessary to match the tensor's dimensionality."**

### Why is this useful?
Consider extracting the last token's representation.
Depending on the model, your tensor might be:
- 3D: `[batch, seq_len, emb_dim]`
- 4D: `[batch, num_heads, seq_len, head_dim]`
- 5D: `[batch, layers, num_heads, seq_len, head_dim]`

Instead of writing custom slicing for each:
```python
# In 3D:
last_step = x[:, -1, :]

# In 4D:
last_step = x[:, :, -1, :]

# Using Ellipsis (...): WORKS FOR BOTH!
last_step = x[..., -1, :]
```

Here, `...` automatically matches all leading dimensions (batch, heads, etc.), focuses on the second-to-last dimension (`-1`), and retains all trailing dimensions (`:`).

---

## 6. Inserting Dimensions On The Fly: `None` / `np.newaxis`

In PyTorch, indexing with `None` is identical to calling `.unsqueeze()`. It inserts a new dimension of size 1 at that exact position.

```python
x = torch.randn(2, 4)  # Shape: [2, 4]

# Insert a new dimension in the middle:
x_expanded = x[:, None, :]
print(x_expanded.shape)
# Output: torch.Size([2, 1, 4])  <-- identical to x.unsqueeze(1)

# Insert a new dimension at the front:
x_batched = x[None, :, :]
print(x_batched.shape)
# Output: torch.Size([1, 2, 4])  <-- identical to x.unsqueeze(0)
```

### LLM Use Case: Attention Mask Broadcasting
```python
# Causal attention mask is 2D: [seq_len, seq_len]
mask = torch.tril(torch.ones(seq_len, seq_len))  # [4, 4]

# To broadcast over [batch_size, num_heads, seq_len, seq_len]:
mask_4d = mask[None, None, :, :]  # Shape: [1, 1, 4, 4]
# Now attention_scores (4D) + mask_4d broadcasts effortlessly!
```

---

## 7. Step-by-Step Walkthrough: The Generation Loop in `ch04.ipynb`

Here is the complete generation function from `ch04.ipynb` annotated with the exact shape changes at every line:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx starts with shape: [batch_size, initial_seq_len] (e.g. [1, 4])

    for _ in range(max_new_tokens):
        # 1. Crop sequence if it exceeds the context window:
        # Slicing keeps [batch_size, current_seq_len <= context_size]
        idx_cond = idx[:, -context_size:]

        # 2. Forward pass through model:
        with torch.no_grad():
            # logits shape: [batch_size, current_seq_len, vocab_size] (e.g. [1, 4, 50257])
            logits = model(idx_cond)

        # 3. Focus only on the prediction from the last token:
        # [:, -1, :] collapses the sequence dimension (dim 1)
        # logits shape becomes: [batch_size, vocab_size] (e.g. [1, 50257])
        logits = logits[:, -1, :]

        # 4. Turn logits into probabilities:
        # probas shape remains: [batch_size, vocab_size]
        probas = torch.softmax(logits, dim=-1)

        # 5. Greedy selection: find token with highest probability:
        # idx_next shape: [batch_size, 1] (because keepdim=True)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 6. Concatenate new token to running sequence along sequence dim (dim=1):
        # [batch_size, current_seq_len] + [batch_size, 1] ==> [batch_size, current_seq_len + 1]
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

---

## 8. Popular Slicing Patterns Across LLMs (BERT, GPT, Attention)

### A. Extracting the `[CLS]` Token (Classification / Embeddings)
In encoder models (BERT, RoBERTa, ViT), the first token (index `0`) represents the sentence/image summary:
```python
# hidden_states: [batch_size, seq_len, emb_dim]
cls_embedding = hidden_states[:, 0, :]  # Shape: [batch_size, emb_dim]
```

### B. Accessing Specific Attention Heads
In Multi-Head Attention, tensors often have shape `[batch, num_heads, seq_len, head_dim]`:
```python
# Inspect attention weights for head 0 only:
# weights: [batch, num_heads, seq_len, seq_len]
head_0_weights = weights[:, 0, :, :]  # Shape: [batch, seq_len, seq_len]
```

### C. Reversing Sequences
```python
# Reverses the token order along sequence dimension (dim 1):
reversed_seq = x[:, ::-1, :]
```

---

## 9. Quick Reference Cheat Sheet

| Expression | Meaning | Input Shape `(B, S, D)` | Output Shape | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `x[:, -1, :]` | Last token representation | `(B, S, D)` | `(B, D)` | Sequence dim **dropped** |
| `x[:, -1:, :]` | Last token with kept dim | `(B, S, D)` | `(B, 1, D)` | Sequence dim **kept** |
| `x[:, 0, :]` | First token (`[CLS]`) | `(B, S, D)` | `(B, D)` | Drops sequence dim |
| `x[:, -K:]` | Last $K$ tokens | `(B, S, D)` | `(B, K, D)` | Context window cropping |
| `x[:, :-1]` | All except last token | `(B, S, D)` | `(B, S-1, D)` | Autoregressive input |
| `x[:, 1:]` | All except first token | `(B, S, D)` | `(B, S-1, D)` | Autoregressive target |
| `x[..., -1]` | Last element of last axis | `(B, S, D)` | `(B, S)` | Drops trailing dim |
| `x[:, None, :]` | Insert dim at index 1 | `(B, D)` | `(B, 1, D)` | Equivalent to `.unsqueeze(1)` |

---

*Keep this reference alongside `01-understanding-torch-dim.md` for complete mastery of PyTorch tensor operations!*

