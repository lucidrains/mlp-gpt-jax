from jax import nn
from jax import random
from jax.lax import top_k
import jax.numpy as np

# helper functions

def exists(val):
    return val is not None

def log(t, eps = 1e-20):
    return np.log(t + eps)

# loss functions

def cross_entropy(logits, targets, axis = -1):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)
    ce = -np.mean(nll)
    return ce

# sampling functions

def select_top_k(tensor, k):
    values, _ = top_k(tensor, k)
    mask = tensor > values.min()
    return mask, np.where(mask, tensor, 0.)

def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -log(-log(noise))

def sample(rng, fn, params, prime, length, top_k = None):
    start_pos = prime.shape[-1]
    seq = np.pad(prime, (0, length - prime.shape[-1]))
    one_hots = np.eye(length, dtype = int)

    for curr_pos in range(start_pos, length):
        logits = fn(params, next(rng), seq)
        logits = logits[curr_pos - 1]

        noise = gumbel_noise(next(rng), logits.shape)

        if exists(top_k):
            mask, logits = select_top_k(logits, top_k)
            noise *= mask

        logits += noise
        sampled_ind = np.argmax(logits, axis = -1)

        one_hot = one_hots[curr_pos]
        seq += one_hot * sampled_ind

    return seq
