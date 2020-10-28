import numpy as np
import tensorflow as tf


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, rng=None):

    if type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def linearly_interpolate(sample, reference=False, num_steps=20):
    if reference is False:
        reference = np.zeros(sample.shape)

    assert sample.shape == reference.shape

    ret = np.zeros(tuple([num_steps+1] + [i for i in sample.shape]))
    for s in range(num_steps+1):
        ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

    return ret.astype(np.float32), num_steps, (sample - reference)


@tf.function
def obtain_gradients(model, inputs):
    with tf.GradientTape() as Tape:
        x = tf.convert_to_tensor(inputs)
        Tape.watch(x)
        output_probs, weights = model(x, training=False)
    dp_dx = Tape.gradient(output_probs, x)
    return dp_dx


def fixed_ig(seq_bag, model, ig_step=20, freq=False):

    reference = []
    if freq:
        reference = np.broadcast_to([0.24, 0.25, 0.25, 0.26], seq_bag.shape)
    else:
        reference = False

    samples, numsteps, step_sizes = linearly_interpolate(seq_bag, reference, ig_step)
    gradients = []
    for i in np.arange(samples.shape[0]):
        tmp = samples[i].reshape((1, seq_bag.shape[0], seq_bag.shape[1], 4))
        gradients.append(obtain_gradients(model, tmp))

    gradients = np.concatenate(gradients, axis=0)
    gradients = (gradients[:-1] + gradients[1:]) / 2
    hyp_scores = np.mean(gradients, axis=0)
    hyp_scores = hyp_scores - np.mean(hyp_scores, axis=-1)[..., np.newaxis]
    ig_scores = np.multiply(hyp_scores, step_sizes)
    return ig_scores, hyp_scores


def dishuffle_ig(seq_bag, model, ref_bag, shuffle_times=20, ig_step=20):
    ig_scores_list = []
    hype_scores_list = []
    for sidx in np.arange(shuffle_times):
        reference = ref_bag[sidx]
        samples, numsteps, step_sizes = linearly_interpolate(seq_bag, reference, ig_step)

        gradients = []
        for i in np.arange(samples.shape[0]):
            tmp = samples[i].reshape((1, seq_bag.shape[0], seq_bag.shape[1], 4))
            gradients.append(obtain_gradients(model, tmp))

        gradients = np.concatenate(gradients, axis=0)
        gradients = (gradients[:-1] + gradients[1:]) / 2
        hyp_scores = np.mean(gradients, axis=0)
        hyp_scores = hyp_scores - np.mean(hyp_scores, axis=-1)[..., np.newaxis]
        ig_scores = np.multiply(hyp_scores, step_sizes)

        ig_scores_list.append(ig_scores[np.newaxis, ...])
        hype_scores_list.append(hyp_scores[np.newaxis, ...])

    mean_scores = np.mean(np.concatenate(ig_scores_list, axis=0), axis=0)
    mean_hype_scores = np.mean(np.concatenate(hype_scores_list, axis=0), axis=0)
    return mean_scores, mean_hype_scores


