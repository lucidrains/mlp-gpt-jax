import jax
from jax import random
from jax import nn
import jax.numpy as np
from functools import partial, wraps
import haiku as hk

# helper functions

def identity(x):
    return x

def residual(fn):
    @wraps(fn)
    def inner(x):
        return fn(x) + x
    return inner

# helpers

LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = True)

# classes

class Identity(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

class gMLP(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        name
    ):
        super().__init__(name = name)
        self.prenorm = LayerNorm(axis = -1)
        self.proj_in = hk.Linear(dim_ff)
        self.proj_out = hk.Linear(dim)
    def __call__(self, x):
        x = self.prenorm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.proj_out(x)
        return x

class MLPGPTJax(hk.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        depth,
        heads = 1,
        ff_mult = 4,
        attn_dim = None,
        clamp_gate = True,
        layer_survival_prob = 1.
    ):
        super().__init__()
        self.embed = hk.Embed(num_tokens, dim)

        self.depth = depth
        self.layer_survival_prob = layer_survival_prob
        self.identity = Identity()
        self.layers = [gMLP(dim = dim, dim_ff = dim * ff_mult, name = f'gmlp{i}') for i in range(depth)]

        self.to_logits = hk.Sequential([
            LayerNorm(axis = -1),
            hk.Linear(num_tokens)
        ])

    def __call__(self, x):
        x = self.embed(x)

        key = hk.next_rng_key()
        should_drop = random.uniform(key, (self.depth,), None, 0., 1.) > self.layer_survival_prob

        for ind, layer in enumerate(self.layers):
            drop_layer = should_drop[ind]
            branch_out = hk.cond(drop_layer, self.identity, layer, x)

        return self.to_logits(x)
