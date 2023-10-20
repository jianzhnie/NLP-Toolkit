# Taken from https://johanwind.github.io/2023/03/23/rwkv_details.html.
# I've added additional comments restructured it a tiny bit, which makes it clearer for me.

import numpy as np
from tokenizers import Tokenizer
from torch import load as torch_load  # Only for loading the model weights

exp = np.exp
layer_norm = lambda x, w, b: (x - np.mean(x)) / np.std(x) * w + b
sigmoid = lambda x: 1 / (1 + exp(-x))


def RWKV(model, token, state):
    params = lambda prefix: [
        model[key] for key in model.keys() if key.startswith(prefix)
    ]

    x = params('emb')[0][token]
    x = layer_norm(x, *params('blocks.0.ln0'))

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3],
                                       *params(f'blocks.{i}.att'))
        x = x + dx

        x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
        dx, state[i][3] = channel_mixing(x_, state[i][3],
                                         *params(f'blocks.{i}.ffn'))
        x = x + dx

    x = layer_norm(x, *params('ln_out'))
    x = params('head')[0] @ x

    e_x = exp(x - np.max(x))
    probs = e_x / e_x.sum()  # Softmax of x

    return probs, state


def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v,
                mix_r, Wk, Wv, Wr, Wout):
    # Part of the state tensor
    #   - last_x  - previous time step embedding (input / prev layer's emb) (1024,)
    #   - last_num - numerator, or "weighted sum of past values" (1024,)
    #   - last_den - denominator, "sum of weights of past values" (1024,)
    # Learnable parameters
    #   - decay (1024,)
    #   - bonus (1024,)
    #   - mix_k - mixing ratio for key (1024,)
    #   - mix_v - mixing ratio for value (1024,)
    #   - mix_r - mixing ratio for receptance (1024,)
    #   - Wk - affine transformation for key (1024, 1024)
    #   - Wv - affine transformation for value (1024, 1024)
    #   - Wr - affine transformation for receptance (1024, 1024)
    #   - Wout - affine transformation for output (1024, 1024)

    # In a typical transformer, the “time mixing” would be done by multi head attention.
    # However, in the RWKV model, the time mixing is done at each time step when
    # num(erator) and den(ominator) are updated. This is similar to how RNNs work.

    # Linear interpolation below between x and last_x uses element-wise mixing ratios
    # mix_*, which are learned weights (of same size as x, last_x).
    # W* are 1024x1024 matrices; matmul with these are most time-consuming.
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    # num / den ~= Weighted average of past values
    # wkv ~= Also weighted average of past values,
    #        but we are adding a "bonus" weight to the current value `v`.
    #        Previous weights get exponentially smaller weight, which is
    #        already captured in the last_num and last_den variables.
    #        However the weight doesn't decay the same for each dimension,
    #        but is determined on each time step based on the decay vector
    #        (see num and den updates below)
    wkv = ((last_num + exp(bonus + k) * v) / (last_den + exp(bonus + k)))
    # Multiplying the wkv (weighted average of past values) with sigmoid(r) is similar
    # to a "gate" in RNNs that controls how much of the past values to use, since
    # sigmoid(r) is a value between 0 and 1.
    rwkv = sigmoid(r) * wkv
    # Final linear (affine) transformation to get the output embedding.
    time_mixed = Wout @ rwkv

    # Below we set the numerator and denominator for the next time step.
    #   num - numerator, or "weighted sum of past values"
    #   den - denominator, "sum of weights of past values"
    # Can be seen as interpolate between previous step num (or den) and a new value,
    # where element-wise decay vector determines the amount of decay per dimension.
    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return time_mixed, (x, num, den)


def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    # Wk - (4096, 1024)
    # Wr - (1024, 1024)
    # Wv - (1024, 4096)

    # In a typical transformer, the “channel mixing” is done by a simple FF NN.
    # By contrast, we use two separate fully connected layers on the input
    # (where input linearly interpolates between the current input and
    # previous time step input) and then multiply them element-wise.

    # Linear interpolation (below) between x and last_x uses an element-wise mixing ratio
    # mix_k and mix_r, which are learned weights (of same size as x, last_x).
    # Wk, Wr, Wv are 1024x1024 matrices; matmul with these are most time-consuming.

    # x and last_x is linearly interpolated with mixing ratio mix_k,
    # then passed through a FC layer with squared relu activation
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))  # @ is matrix multiplication
    k = np.maximum(k, 0)**2  # squared relu activation

    # x and last_x is linearly interpolated with mixing ratio mix_r,
    # then passed through a FC layer with sigmoid activation
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))
    r = sigmoid(r)

    # K-mixed input is passed through affine transformation (without activation,
    # so not quite a FC layer) before being multiplied to r-mixed input element-wise.
    vk = Wv @ k
    channel_mixed = r * vk

    return channel_mixed, x  # pass x along unchanged, will be last_x in the next step


def sample_probs(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs**(1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))


# Available at https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth
MODEL_FILE = 'data/rwkv/RWKV-4-Pile-430M-20220808-8066.pth'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}')
weights = torch_load(MODEL_FILE, map_location='cpu')
for k in weights.keys():
    if '.time_' in k:
        weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy()  # convert to f32 type

# Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
tokenizer = Tokenizer.from_file('data/rwkv/20B_tokenizer.json')

print('\nPreprocessing context')
context = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, \
    previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

# The 4 dimensions are
#     [last_x, last_num, last_den] (after time mixing) - used by time mixing
#     last_x (after channel mixing) - used by channel mixing
state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

print(context, end='')
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end='', flush=True)
    probs, state = RWKV(weights, token, state)
