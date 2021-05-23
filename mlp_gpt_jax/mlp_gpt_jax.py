import jax
from jax import numpy as np
import jax.nn.initializers as init

from flax import linen as nn
from einops import rearrange

class gMLP(nn.Module):
    dim: int
    seq_len: int
    dim_ff: int
    heads: int
    clamp_gate: bool

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.dim_ff)(x)
        x = nn.gelu(x)
        x = SGU(seq_len = self.seq_len, heads = self.heads, clamp_gate = self.clamp_gate)(x)
        x = nn.Dense(features = self.dim)(x)
        return x

class SGU(nn.Module):
    seq_len: int
    heads: int
    clamp_gate: bool

    @nn.compact
    def __call__(self, x):
        n, h = self.seq_len, self.heads
        x, gate = np.split(x, 2, axis = -1)
        gate = nn.LayerNorm()(gate)

        weights = self.param('spatial_weights', init.uniform(scale = 1e-3 // n), (h, n, n))
        bias = self.param('spatial_bias', init.ones, (n, h, 1))

        mask = np.tril(np.ones((n, n)))
        weights = weights * rearrange(mask, 'm n -> () m n')

        gate = rearrange(gate, 'n (h d) -> n h d', h = h)
        gate = np.einsum('n h d, h m n -> m h d', gate, weights)
        gate = gate + bias
        gate = rearrange(gate, 'n h d -> n (h d)')

        gate = np.tanh(gate) if self.clamp_gate else gate
        return x * gate

class MLPGpt(nn.Module):
    num_tokens: int
    dim: int
    seq_len: int
    depth: int
    heads: int = 1
    ff_mult: int = 4
    clamp_gate: bool = True

    def setup(self):
        self.layers = [gMLP(dim = self.dim, dim_ff = self.dim * self.ff_mult, seq_len = self.seq_len, heads = self.heads, clamp_gate = self.clamp_gate) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        to_embed = nn.Embed(num_embeddings = self.num_tokens, features = self.dim)
        x = to_embed(x)

        for layer in self.layers:
            x = layer(x) + x

        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.num_tokens)(x)
        return x
