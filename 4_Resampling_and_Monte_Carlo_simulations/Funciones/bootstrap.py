import numpy as np


# Bootstrap replicate for one dimensional array

def bootstrap_replicate_1d(data, func):
    """ Compute a single value of a statistc. """    
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


# Generating many bootstrap replicates

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Generate replicates
    bs_replicates = [bootstrap_replicate_1d(data, func) for i in range(size)]

    return np.asarray(bs_replicates)
