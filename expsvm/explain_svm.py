from math import comb
import numpy as np
import itertools as it
from typing import List, Tuple


class TensorPerm:
    def __init__(self, rank: int, dim: int) -> None:
        self.rank = rank
        self.dim = dim
        self.idx_unique = None
        self.idx_n_perm = None

    def create_unique_index(self) -> List[Tuple]:
        """
        Create set of tensor indexes where the order of the indexes does not matter.
        """
        return list(it.combinations_with_replacement(np.arange(self.dim), self.rank))

    def _count_index_occurrences(self):
        """
        Count the number of occurrences of each index, e.g. the element (0,0,1,2,3) has occurrences {2,1,1,1,0}
        corresponding to two 0s, one 1, one 2, one 3 and zero 4. The 4 ins included since this is the highest possible
        index in this tensor.
        Also, map each occurrence list to a label.
        """
        counts_unique = set()
        elem_counts = []
        for elem in self.idx_unique:
            idx_set = set(elem)
            counts = []
            for idx in idx_set:
                count = elem.count(idx)
                counts.append(count)
            counts.sort()
            counts = ','.join([str(sorted_count) for sorted_count in counts])
            counts_unique.add(counts)
            elem_counts.append(counts)
        return elem_counts, list(counts_unique)

    def n_perm(self) -> None:
        self.idx_unique = self.create_unique_index()
        count_strs, count_strs_unique = self._count_index_occurrences()
        # Map number of permutations to each unique index in idx_unique.
        perm_count_dict = {count_str: self._count_perm(count_str) for count_str in count_strs_unique}
        self.idx_n_perm = list(map(perm_count_dict.get, count_strs))

    def _count_perm(self, count_str: str) -> int:
        """
        Calculate number of symmetries from a string formatted as 'xyz...' where x is the number of occurrences of the
        first index, y is the number of occurrences of the second index, etc.
        Example:
        A tensor element is located at (i,j,i,k,l). The number of permutations, e.g. (i,i,j,k,l),
        is then 5!/(2!)=60
        5! is from the rank of the tensor. It's possible to place 5 distinct values in 5! ways.
        Now, 2 indexes were the same, meaning the total number of unique permutations is reduced.
        The reduction is 2!, from the number of possible ways we can interchange the two identical indexes.
        Rank of tensor is 5=2+1+1+1, i.e. the number of occurences of i,j,k,l.

        :param count_str: String of occurrences of each index.
        :return: int. Number of possible permutations
        """
        counts_list = [int(ch) for ch in count_str.split(',') if int(ch)>1]
        if len(counts_list) == 0:
            perm_reduction = 1
        else:
            perm_reduction = np.prod([np.math.factorial(count) for count in counts_list])
        n_perm = int(np.math.factorial(self.rank)/perm_reduction)

        return n_perm


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
        self.signed_alpha = alpha*class_label

        # Polynomial kernel parameters
        self.kernel_d = kernel_d
        self.kernel_r = kernel_r

        # Instantiate compressed polynomial SVM
        self.idx_unique = {ind: None for ind in np.arange(1, self.kernel_d + 1)}
        self.sym_count = {ind: None for ind in np.arange(1, self.kernel_d + 1)}
        self.poly_coeff = {ind: None for ind in np.arange(1, self.kernel_d + 1)}

    def _multiplication_transform(self) -> None:
        for d in np.arange(1, self.kernel_d + 1):
            tp = TensorPerm(d, self.p)
            tp.n_perm()
            self.idx_unique[d] = np.array(tp.idx_unique)
            self.sym_count[d] = np.array(tp.idx_n_perm)

