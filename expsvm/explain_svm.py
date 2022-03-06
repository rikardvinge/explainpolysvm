from math import comb
import numpy as np
import itertools as it
from typing import List, Tuple


def create_unique_index(p: int, tensor_rank: int = None) -> List[Tuple]:
    """
    Create set of tensor indexes where the order of the indexes does not matter.
    :param p: dimensionality of original space
    :param tensor_rank: Rank of tensor
    """
    if tensor_rank is None:
        tensor_rank = p
    return list(it.combinations_with_replacement(np.arange(p), tensor_rank))


def count_index_occurrences(elems: List[Tuple]):
    """
    Count the number of occurrences of each index, e.g. the element (0,0,1,2,3) has occurrences {2,1,1,1,0}
    corresponding to two 0s, one 1, one 2, one 3 and zero 4. The 4 ins included since this is the highest possible
    index in this tensor.
    Also, map each occurrence list to a label.

    :param elems: List of index-tuples describing the location of an element of a tensor.
    """
    counts_unique = set()
    elem_counts = []
    for elem in elems:
        unique_idxs = set(elem)
        counts = []
        for idx in unique_idxs:
            count = elem.count(idx)
            counts.append(count)
        counts.sort()
        counts = ''.join([str(sorted_count) for sorted_count in counts])
        counts_unique.add(counts)
        elem_counts.append(counts)
    return elem_counts, list(counts_unique)


def count_symmetry(count_str: str, tensor_rank: int = None) -> int:
    """
    Calculate number of symmetries from a string formatted as 'xyz...' where x is the number of occurrences of the
    first index, y is the number of occurrences of the second index, etc.
    Example:
    A tensor element is located at (i,j,i,k,l). The number of symmetries, e.g. (i,i,j,k,l), is then
    First index i have two occurrences and thus (5 over 2) possible placements.
    Second index j can occur at (3 over 1) places. 3 comes from 5-2, the tensor dimension minus the number of i.
    Third index k can occur at (2 over 1) places.
    Final index l can only occur at one place.
    Total number of permutations: (5 over 2)(3 over 1)(2 over 1)1.
    Rank of tensor is 5=2+1+1+1, i.e. the number of occurences of i,j,k,l.

    :param count_str: String of occurrences of each index.
    :param tensor_rank: (Optional) rank of tensor.
    :return: int. Number of
    """
    counts_list = [int(ch) for ch in count_str]
    nperm = 1
    if tensor_rank is None:
        dims_left = sum(counts_list)
    else:
        dims_left = tensor_rank
    for count in counts_list:
        nperm *= comb(dims_left, count)
        dims_left -= count

    return nperm


def map_symmetry(count_strs: List[str], count_strs_unique: List[str] = None):
    """
    Map the number of symmetries for each count_str in count_strs.

    :param count_strs: List of strings of index occurrences.
    :param count_strs_unique: (Optional) Unique index occurrences. Use to speed up calculation.
    :return:
    """
    if count_strs_unique is None:
        count_strs_unique = set(count_strs)

    sym_count_dict = {count_str: count_symmetry(count_str) for count_str in count_strs_unique}
    sym_count = list(map(sym_count_dict.get, count_strs))
    return sym_count


class ExPSVM:
    def __init__(self, sv: np.ndarray, alpha: np.array, class_label: np.ndarray, kernel_d: int, kernel_r: float,
                 p: int = None) -> None:
        # Number of features
        if p is None:
            self.p = sv.shape[0]
        else:
            self.p = p

        # SVM model
        self.sv = sv
        self.alpha = alpha
        self.label = class_label

        # Polynomial kernel parameters
        self.kernel_d = kernel_d
        self.kernel_r = kernel_r

        # Instantiate compressed polynomial SVM
        self.unique_idx = {ind: None for ind in np.arange(1, self.kernel_d + 1)}
        self.sym_count = {ind: None for ind in np.arange(1, self.kernel_d + 1)}

    def dekernelize(self):
        self.unique_idx[1] = np.ones(self.p)
        self.sym_count[1] = np.ones(self.p)

        # For higher than 2 dimensional polynomial kernels.
        if self.kernel_d > 1:
            for d in np.arange(2, self.kernel_d + 1):
                idx = create_unique_index(self.p, d)
                labels, sym_count_unique = count_index_occurrences(idx)
                sym_count = map_symmetry(labels, sym_count_unique)
                self.unique_idx[d] = np.array(idx)
                self.sym_count[d] = np.array(sym_count)
        return
