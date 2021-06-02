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
from haiku import transform
from mlp_gpt_jax import MLPGpt

@transform
def model(seq):
    return MLPGpt(
        num_tokens = 20000,
        dim = 512,
        depth = 6,
        seq_len = 1024
    )(seq)

key = random.PRNGKey(0)
seq = random.randint(key, (1024,), 0, 20000)

params = model.init(key, seq)
logits = model.apply(params, key, seq) # (1024, 20000)
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
