from random import randrange
import tqdm
import gzip
import numpy as np

from torch.utils.data import DataLoader, Dataset

import jax
from jax import nn
from jax import value_and_grad, vmap, jit, random
from optax import adam, clip_by_global_norm, chain, apply_updates, apply_every

from haiku import PRNGSequence

from mlp_gpt_jax import TransformedMLPGpt
from mlp_gpt_jax.utils import sample, cross_entropy

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.5
VALIDATE_EVERY  = 100
SAMPLE_EVERY  = 500
SEQ_LEN = 768

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    data_train, data_val = np.split(X, [int(90e6)])

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = randrange(0, self.data.shape[0] - self.seq_len - 1)
        return self.data[rand_start: rand_start + self.seq_len + 1]

    def __len__(self):
        return self.data.shape[0] // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# setup model and params

model_kwargs = dict(
    num_tokens = 256,
    dim = 512,
    seq_len = SEQ_LEN,
    depth = 8,
    attn_dim = 32,
)

train_model = TransformedMLPGpt(**model_kwargs, layer_survival_prob = 0.95)
eval_model = TransformedMLPGpt(**model_kwargs)

rng = PRNGSequence(42)
params = train_model.init(next(rng), train_dataset[0][:-1])

# loss function

batch_model_apply = jit(vmap(train_model.apply, in_axes = (None, None, 0), out_axes = 0))

@value_and_grad
def loss_fn(params, key, data):
    inp, labels = data[:, :-1], data[:, 1:]
    logits = batch_model_apply(params, key, inp)
    return cross_entropy(logits, labels, axis = -1)

# optimizer

optim = chain(
    clip_by_global_norm(MAX_GRAD_NORM),
    adam(LEARNING_RATE),
    apply_every(GRADIENT_ACCUMULATE_EVERY)
)

optim_state = optim.init(params)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    data = next(train_loader).numpy()
    loss, grads = loss_fn(params, next(rng), data)
    updates, optim_state = optim.update(grads, optim_state, params)
    params = apply_updates(params, updates)

    if i % GRADIENT_ACCUMULATE_EVERY == 0:
        print(f'loss: {loss.item()}')

    if i % SAMPLE_EVERY == 0:
        valid_data = next(val_loader).numpy()
        prime = valid_data[0][:100]
        prime_str = decode_tokens(prime)
        print(prime_str, "\n", "*" * 40)

        sampled = sample(rng, jit(eval_model.apply), params, prime, SEQ_LEN, top_k = 25)
        sampled_str = decode_tokens(sampled[100:])
        print(sampled_str)
