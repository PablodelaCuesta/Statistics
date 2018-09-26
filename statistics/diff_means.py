import numpy as np

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.abs(np.mean(data_1) - np.mean(data_2))

    return diff
