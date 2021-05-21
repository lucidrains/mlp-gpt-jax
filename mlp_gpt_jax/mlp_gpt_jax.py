import jax
from jax import numpy as np
import jax.nn.initializers as init

from flax import linen as nn
from einops import rearrange

class gMLP(nn.Module):
    dim: int
    seq_len: int
    dim_ff: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.dim_ff)(x)
        x = nn.gelu(x)
        x = SGU(seq_len = self.seq_len)(x)
        x = nn.Dense(features = self.dim)(x)
        return x

class SGU(nn.Module):
    seq_len: int

    def setup(self):
        return

    @nn.compact
    def __call__(self, x):
        n = self.seq_len
        x, gate = np.split(x, 2, axis = -1)
        gate = nn.LayerNorm()(gate)

        weights = self.param('spatial_weights', init.uniform(scale = 1e-3 // n), (n, n))
        bias = self.param('spatial_bias', init.ones, (n, 1))

        tri_mask = np.ones((n, n))
        causal_mask = np.triu(tri_mask, 1)
        weights *= causal_mask

        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate = gate + bias

        return x * gate

class MLPGpt(nn.Module):
    num_tokens: int
    dim: int
    seq_len: int
    depth: int
    ff_mult: int = 4

    def setup(self):
        self.layers = [gMLP(dim = self.dim, dim_ff = self.dim * self.ff_mult, seq_len = self.seq_len) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        to_embed = nn.Embed(num_embeddings = self.num_tokens, features = self.dim)
        x = to_embed(x)

        for layer in self.layers:
            x = layer(x) + x

        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.num_tokens)(x)
        return x
