## MLP GPT - Jax

A GPT, made only of MLPs, in Jax. The specific MLP to be used are <a href="https://arxiv.org/abs/2105.08050">gMLPs</a> with the Spatial Gating Units.

<a href="https://github.com/lucidrains/g-mlp-gpt">Working Pytorch implementation</a>

## Install

```bash
$ pip install mlp-gpt-jax
```

## Usage

```python
from jax import random
from haiku import PRNGSequence
from mlp_gpt_jax import TransformedMLPGpt

model = TransformedMLPGpt(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1024
)

rng = PRNGSequence(0)
seq = random.randint(next(rng), (1024,), 0, 20000)

params = model.init(next(rng), seq)
logits = model.apply(params, next(rng), seq) # (1024, 20000)
```

To use the tiny attention (also made autoregressive with a causal mask), just set the `attn_dim` to the head dimension you'd like to use. `64` was recommended in the paper

```python
from jax import random
from haiku import PRNGSequence
from mlp_gpt_jax import TransformedMLPGpt

model = TransformedMLPGpt(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1024,
    attn_dim = 64     # set this to 64
)

rng = PRNGSequence(0)
seq = random.randint(next(rng), (1024,), 0, 20000)

params = model.init(next(rng), seq)
logits = model.apply(params, next(rng), seq) # (1024, 20000)
```

## Citations

```bibtex
@misc{liu2021pay,
    title   = {Pay Attention to MLPs}, 
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
