import jax
from jax import numpy as np
import jax.nn.initializers as init

from flax import linen as nn
from einops import rearrange

# constants

ATTN_MASK_VALUE = -1e10

# helper functions

def exists(val):
    return val is not None

# classes

class gMLP(nn.Module):
    dim: int
    seq_len: int
    dim_ff: int
    heads: int
    clamp_gate: bool
    attn_dim: int = None

    def setup(self):
        self.attn = Attention(dim = self.dim, dim_head = self.attn_dim, dim_out = self.dim_ff // 2, seq_len = self.seq_len) if exists(self.attn_dim) else None

    @nn.compact
    def __call__(self, x_):
        x = nn.LayerNorm()(x_)
        x = nn.Dense(features = self.dim_ff)(x)
        x = nn.gelu(x)

        gate_res = self.attn(x_) if exists(self.attn) else None

        x = SGU(seq_len = self.seq_len, heads = self.heads, clamp_gate = self.clamp_gate)(x, gate_res)
        x = nn.Dense(features = self.dim)(x)
        return x

class Attention(nn.Module):
    dim: int
    dim_head: int
    dim_out: int
    seq_len: int

    @nn.compact
    def __call__(self, x):
        n = self.seq_len
        scale = self.dim_head ** -0.5
        qkv = nn.Dense(features = self.dim_head * 3, use_bias = False)(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        sim = np.einsum('i d, j d -> i j', q, k) * scale

        mask = np.tril(np.ones((n, n)))
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        attn = nn.softmax(sim, axis = -1)
        out = np.einsum('i j, j d -> i d', attn, v)
        out = nn.Dense(features = self.dim_out)(out)
        return out

class SGU(nn.Module):
    seq_len: int
    heads: int
    clamp_gate: bool

    @nn.compact
    def __call__(self, x, gate_res = None):
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

        if exists(gate_res):
            gate = gate + gate_res

        gate = np.tanh(gate) if self.clamp_gate else gate
        return x * gate

class MLPGpt(nn.Module):
    num_tokens: int
    dim: int
    seq_len: int
    depth: int
    heads: int = 1
    ff_mult: int = 4
    attn_dim: int = None
    clamp_gate: bool = True

    def setup(self):
        self.layers = [gMLP(dim = self.dim, dim_ff = self.dim * self.ff_mult, seq_len = self.seq_len, heads = self.heads, clamp_gate = self.clamp_gate, attn_dim = self.attn_dim) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        to_embed = nn.Embed(num_embeddings = self.num_tokens, features = self.dim)
        x = to_embed(x)

        for layer in self.layers:
            x = layer(x) + x

        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.num_tokens)(x)
        return x
