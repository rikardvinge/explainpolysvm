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
        list_idx = list(it.combinations_with_replacement(np.arange(self.dim), self.rank))
        return [','.join([str(idx) for idx in tup]) for tup in list_idx]

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
            idx_set = set(elem.split(','))
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
    def __init__(self, sv: np.ndarray, dual_coeff: np.ndarray, kernel_d: int, kernel_r: float,
                 p: int = None) -> None:
        # Number of features
        if p is None:
            self.p = sv.shape[0]
        else:
            self.p = p

        # SVM model
        self.sv = sv
        self.dual_coef = np.reshape(dual_coeff,(-1,1))

        # Polynomial kernel parameters
        self.kernel_d = kernel_d
        self.kernel_r = kernel_r

        # Instantiate compressed polynomial SVM
        # self.idx_unique = dict()
        self.idx_unique = []
        # self.perm_count = dict()
        self.perm_count = np.array([])#[]
        self.idx_dim = np.array([])#[]
        self.poly_coef = {d: comb(self.kernel_d, d) * (self.kernel_r ** (self.kernel_d - d)) for d in np.arange(1, self.kernel_d + 1)}

    def _multiplication_transform(self) -> None:
        """
        For each polynomial term, identify unique indexes and the number of permutations of each index.
        """
        for d in np.arange(1, self.kernel_d + 1):
            tp = TensorPerm(d, self.p)
            tp.n_perm()
            self.idx_unique = np.concatenate((self.idx_unique,tp.idx_unique))
            self.perm_count = np.concatenate((self.perm_count,tp.idx_n_perm))
            self.idx_dim = np.concatenate((self.idx_dim,d*np.ones((len(tp.idx_unique),))))
            # self.idx_unique[d] = np.array(tp.idx_unique)
            # self.perm_count[d] = np.array(tp.idx_n_perm)
            # self.perm_count = np.array(self.perm_count)

    def _compress_transform(self, x: np.ndarray, memory_optimize: bool = False, to_array: bool = False):
        """
        :param: x: Observation(s) to transform. Shape of ndarray is assumed to be n_observations-by-n_features.
        :param memory_optimize: Set to False (default) for handling all support vectors and transformations at once. If True, handle one support vector at a time for reduced memory usage.
        :param to_array: Set to True if output should be array instead of dict.
        :return:
        """
        transf = dict()
        for d in np.arange(1, self.kernel_d + 1):
            d_idx = [[int(ch) for ch in idx.split(',')] for idx in self.idx_unique[self.idx_dim==d]]
            if memory_optimize:
                sv_interaction = np.zeros((x.shape[0],len(d_idx)))
                for ind, v in enumerate(x):
                    sv_interaction[ind,:] = np.squeeze(np.prod(v[d_idx], axis=1))
            else:
                sv_interaction = np.squeeze(np.prod(x[:,d_idx], axis=2))
            transf[d] = sv_interaction
        if to_array:
            return self.dict2array(transf)
        else:
            return transf

    @classmethod
    def dict2array(cls, di) -> np.ndarray:
        """
        Flatten dict to array. Dict is assumed to have structure {1:array_1, 2:array_2,...} where array_x are 2d arrays where rows are assumed belonging together.
        Example: The array
        {1:np.array([[1, 2, 3], [1, 2, 3]]),2:np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
        is returned as [[1, 2, 3, 1, 2, 3, 4, 6, 9], [1, 2, 3, 1, 2, 3, 4, 6, 9]]

        :param di: Dictionary to transform
        :return:
        """
        return np.concatenate([*di.values()], axis=1)


    def dekernelize(self, memory_optimize: bool = False):
        """
        Calculate compressed linear version for SVM model with polynomial kernel.
        :param memory_optimize:
        :return:
        """
        if not ((self.idx_unique) and (self.perm_count)):
            self._multiplication_transform()
        compressed_transform = self._compress_transform(x=self.sv, memory_optimize=memory_optimize, to_array=False)

        # Multiply by dual coefficients (alpha and labels), sum over support vectors, and scale with polynomial coefficient
        for d in compressed_transform.keys():
            compressed_transform[d] = self.poly_coef[d]*np.sum(np.multiply(self.dual_coef, compressed_transform[d]), axis=0)

        # Create linear model
        print()
        print(compressed_transform.values())
        lin = np.concatenate(list(compressed_transform.values()))
        print(lin)
        print(lin.shape)