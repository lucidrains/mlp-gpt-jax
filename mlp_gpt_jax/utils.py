import haiku as hk
from jax import random
from jax.lax import top_k
import jax.numpy as np

# loss functions

def cross_entropy(logits, targets, axis = -1):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = np.take_along_axis(logprobs, np.expand_dims(targets, axis = axis), axis = axis)
    ce = -np.mean(nll)
    return ce

# sampling functions

def select_top_k(tensor, k):
    val, _ = top_k(tensor, k)
    thres = val.min()
    return np.where(tensor > thres, tensor, 0.)

def gumbel_noise(rng, shape):
    noise = random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
    return -np.log(-np.log(noise))

def sample(rng, fn, params, prime, length, top_k = None):
    start_pos = prime.shape[-1]
    seq = np.pad(prime, (0, length - prime.shape[-1]))
    one_hots = np.eye(length, dtype = int)

    for curr_pos in range(start_pos, length):
        logits = fn(params, next(rng), seq)
        logits = logits[curr_pos - 1]

        if top_k is not None:
            logits = select_top_k(logits, top_k)

        logits += gumbel_noise(next(rng), logits.shape)
        sampled_ind = np.argmax(logits, axis = -1)
        one_hot = one_hots[curr_pos]
        seq += one_hot * sampled_ind

    return seq
