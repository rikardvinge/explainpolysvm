import numpy as np
import itertools as it
from typing import List


def create_tensor_index(n: int, r: int = None):
    """
    Create set of tensor indexes where the order of the indexes does not matter.
    :param n: Dimension of the tensor
    :param r: Number of indexes to create
    """
    if r is None:
        r = n
    return it.combinations_with_replacement(np.arange(n), r)


def repeated_elements(n_ids: List[int]):
    """

    :param n_ids: List of number of repeats of each index
    """

    return n_ids