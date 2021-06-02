from functools import partial

import jax
from jax import random
from jax import nn
import jax.numpy as np

import haiku as hk
from einops import rearrange

# constants

EPS = 1e-3

# helpers

LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = True, axis = -1)

# classes

class SGU(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        self.norm = LayerNorm()
        self.proj_out = hk.Linear(dim_out)

    def __call__(self, x):
        n = self.seq_len
        x, gate = np.split(x, 2, axis = -1)

        gate = self.norm(gate)

        init_eps = hk.initializers.Constant(EPS / n)

        weights = hk.get_parameter('spatial_weights', shape = (n, n), init = init_eps)
        biases = hk.get_parameter('spatial_biases', shape = (n,), init = np.ones)

        mask = np.tril(np.ones((n, n)))
        weights = weights * mask

        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate = gate + rearrange(biases, 'n -> n ()')

        x = x * gate
        return self.proj_out(x)

class gMLP(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        name
    ):
        super().__init__(name = name)
        self.norm = LayerNorm()
        self.proj_in = hk.Linear(dim_ff)
        self.sgu = SGU(dim = dim_ff, dim_out = dim_ff // 2, seq_len = seq_len)
        self.proj_out = hk.Linear(dim)

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x

class MaybeExecute(hk.Module):
    def __init__(
        self,
        *,
        prob_execute,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.prob_execute = prob_execute

    def __call__(self, x):
        key = hk.next_rng_key()
        p = random.bernoulli(key, p = self.prob_execute)
        out = self.fn(x) * p + 0 * (1 - p)
        return out / self.prob_execute

class MLPGpt(hk.Module):
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

        gmlps = [gMLP(dim = dim, dim_ff = dim * ff_mult, seq_len = seq_len, name = f'gmlp{i}') for i in range(depth)]
        self.layers = [MaybeExecute(prob_execute = layer_survival_prob, fn = gmlp) for gmlp in gmlps]

        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_tokens)
        ])

    def __call__(self, x):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x) + x

        return self.to_logits(x)
