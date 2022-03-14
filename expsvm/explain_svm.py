from math import comb
import numpy as np
import itertools as it
from typing import List


class TensorUtil:
    def __init__(self, rank: int, dim: int) -> None:
        self.rank = rank
        self.dim = dim

    def create_unique_index(self) -> List[str]:
        """
        Create set of tensor indexes where the order of the indexes does not matter.
        """
        list_idx = list(it.combinations_with_replacement(np.arange(self.dim), self.rank))
        return [','.join([str(idx) for idx in tup]) for tup in list_idx]

    @classmethod
    def _count_index_occurrences(cls, idx_list):
        """
        Count the number of occurrences of each index, e.g. the element (0,0,1,2,3) has occurrences {2,1,1,1,0}
        corresponding to two 0s, one 1, one 2, one 3 and zero 4. The 4 ins included since this is the highest possible
        index in this tensor.
        Also, map each occurrence list to a label.
        """
        counts_unique = set()
        elem_counts = []
        for elem in idx_list:
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

    def n_perm(self) -> (List[str], List[int]):
        idx_unique = self.create_unique_index()
        count_strs, count_strs_unique = self._count_index_occurrences(idx_unique)
        # Map number of permutations to each unique index in idx_unique.
        perm_count_dict = {count_str: self._count_perm(count_str) for count_str in count_strs_unique}
        idx_n_perm = list(map(perm_count_dict.get, count_strs))
        return idx_unique, idx_n_perm

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
        counts_list = [int(ch) for ch in count_str.split(',') if int(ch) > 1]
        if len(counts_list) == 0:
            perm_reduction = 1
        else:
            perm_reduction = np.prod([np.math.factorial(count) for count in counts_list])
        n_perm = int(np.math.factorial(self.rank) / perm_reduction)

        return n_perm


class ExPSVM:

    def __init__(self, sv: np.ndarray, dual_coeff: np.ndarray,
                 intercept: float,
                 kernel_d: int, kernel_r: float,
                 p: int = None) -> None:
        # Number of features
        if p is None:
            self.p = sv.shape[0]
        else:
            self.p = p

        # SVM model
        self.sv = sv
        self.dual_coef = np.reshape(dual_coeff, (-1, 1))
        self.intercept = intercept

        # Polynomial kernel parameters
        self.kernel_d = kernel_d
        self.kernel_r = kernel_r

        # Instantiate compressed polynomial SVM
        self.idx_unique = np.array([])
        self.perm_count = np.array([])
        self.idx_dim = np.array([])
        self.poly_coef = {d: comb(self.kernel_d, d) * (self.kernel_r ** (self.kernel_d - d))
                          for d in np.arange(1, self.kernel_d + 1)}
        self.mask_idx = np.array([])

        self.linear_model = np.array([])

    def get_idx_unique(self, **kwargs) -> np.ndarray:
        return self.idx_unique[self.get_dim_mask_index(**kwargs)]

    def get_perm_count(self, **kwargs) -> np.ndarray:
        return self.perm_count[self.get_dim_mask_index(**kwargs)]

    def get_idx_dim(self, **kwargs) -> np.ndarray:
        return self.idx_dim[self.get_dim_mask_index(**kwargs)]

    def get_linear_model(self, **kwargs) -> np.ndarray:
        return self.linear_model[self.get_dim_mask_index(**kwargs), :]

    def _multiplication_transform(self) -> None:
        """
        For each polynomial term, identify unique indexes and the number of permutations of each index.
        """
        for d in np.arange(1, self.kernel_d + 1):
            tp = TensorUtil(d, self.p)
            idx_tmp, n_perm_tmp = tp.n_perm()
            self.idx_unique = np.concatenate((self.idx_unique, idx_tmp))
            self.perm_count = np.concatenate((self.perm_count, n_perm_tmp))
            self.idx_dim = np.concatenate((self.idx_dim, d * np.ones((len(idx_tmp),))))

    def _compress_transform(self, x: np.ndarray, reduce_memory: bool = False, mask: bool = False):
        """
        :param: x: Observation(s) to transform. Shape of ndarray is assumed to be n_observations-by-n_features.
        :param reduce_memory: Set to False (default) for handling all support vectors and transformations at once. If
        True, handle one support vector at a time for reduced memory usage.
        :return: trans_dict, the transformed features of x. Structured as dict with
        keys being the polynomial degree. The order is the same as given by get_dim_idx(d, mask)
        """
        trans_dict = dict()
        for d in np.arange(1, self.kernel_d + 1):
            d_idx = [[int(ch) for ch in idx.split(',')] for idx in self.get_idx_unique(d=d, mask=mask)]
            if len(d_idx) > 0:
                if reduce_memory:
                    feat_trans = np.zeros((x.shape[0], len(d_idx)))
                    for ind, v in enumerate(x):
                        feat_trans[ind, :] = np.squeeze(np.prod(np.expand_dims(v, axis=0)[:, d_idx], axis=2))
                else:
                    feat_trans = np.squeeze(np.prod(x[:, d_idx], axis=2))
                if len(feat_trans.shape) == 1:
                    feat_trans = np.reshape(feat_trans, (1, feat_trans.shape[0]))
                trans_dict[d] = feat_trans
            else:
                trans_dict[d] = np.array([])
        return trans_dict

    def get_dim_mask_index(self, d: int = None, mask: bool = False) -> np.ndarray:
        """
        Returns index in transformed space of dimension d that are in mask.

        :param d: Dimension to extract indexes from.
        :param mask: Set to True to extract only indexes in mask of dimension d. Default False.
        :return:
        """
        if d is None:
            ind = np.full(self.idx_unique.shape, True)
        else:
            ind = self.idx_dim == d
        if mask:
            if self.mask_idx.size > 0:
                ind = ind & self.mask_idx
            else:
                raise ValueError("Instance variable mask is empty.")
        return ind

    @classmethod
    def dict2array(cls, di) -> np.ndarray:
        """
        Flatten dict to array. Dict is assumed to have structure {1:array_1, 2:array_2,...} where array_x are 2d
        arrays where rows are assumed belonging together.

        Example: The array
        {1:np.array([[1, 2, 3], [1, 2, 3]]),2:np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
        is returned as [[1, 2, 3, 1, 2, 3, 4, 6, 9], [1, 2, 3, 1, 2, 3, 4, 6, 9]]

        :param di: Dictionary to transform
        :return: numpy.ndarray of shape n_sample-by-n_transformed_features-shaped.
        """
        # return np.concatenate([*di.values()], axis=-1)
        return np.concatenate([*{key: val for key, val in di.items() if val.size > 0}.values()], axis=-1)

    def transform_svm(self, reduce_memory: bool = False, mask: bool = False):
        """
        Calculate compressed linear version for SVM model with polynomial kernel.
        :param mask:
        :param reduce_memory:
        :return:
        """
        if (self.idx_unique.size == 0) or (self.perm_count.size == 0):
            self._multiplication_transform()
        compressed_transform = self._compress_transform(x=self.sv, reduce_memory=reduce_memory, mask=mask)

        # Multiply by dual coefficients, sum over support vectors, and scale with polynomial coefficient
        for d in compressed_transform.keys():
            if compressed_transform[d].size > 0:
                compressed_transform[d] = self.poly_coef[d] * (
                    np.sum(np.multiply(self.dual_coef,
                                       np.multiply(self.get_perm_count(d=d, mask=mask), compressed_transform[d])),
                           axis=0, keepdims=False))

        # Create linear model
        self.linear_model = np.expand_dims(self.dict2array(compressed_transform), axis=1)

    def polynomial_kernel(self, x: np.ndarray, y: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(x.shape) > 2:
            raise ValueError("x should be 2-dimensional. Shape of x is {}".format(x.shape))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))
        if len(y.shape) > 2:
            raise ValueError("y should be 2-dimensional. Shape of y is {}".format(y.shape))
        return (self.kernel_r + np.sum(np.multiply(x, y), axis=1, keepdims=True)) ** self.kernel_d

    def _linear_model_dot(self, x: np.ndarray, reduce_memory: bool = False, mask: bool = False):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(x.shape) > 2:
            raise ValueError("x should be 2-dimensional. Shape of x is {}".format(x.shape))
        x_trans = self.dict2array(self._compress_transform(x=x, reduce_memory=reduce_memory, mask=mask))
        dot_prod = np.matmul(x_trans, self.get_linear_model(mask=mask))
        return dot_prod + self.kernel_r ** self.kernel_d

    def decision_function(self, x: np.ndarray, reduce_memory: bool = False, mask: bool = False):
        return self.intercept + self._linear_model_dot(x=x, reduce_memory=reduce_memory, mask=mask)

    def feature_importance(self, sort: bool = True, **kwargs):
        if sort:
            sort_order = np.argsort(np.squeeze(self.get_linear_model(**kwargs)))[::-1]
        else:
            sort_order = np.arange(self.idx_unique.size)
        feat_imp = np.abs(self.get_linear_model(**kwargs)[sort_order, 0])
        feat = self.get_idx_unique(**kwargs)[sort_order]
        return feat_imp, feat, sort_order

    def feature_selection(self, n_feat: int = None,
                          frac_feat: float = None,
                          frac_feat_imp: float = None):
        """

        :param n_feat:
        :param frac_feat:
        :param frac_feat_imp:
        :return:
        """

        feat_imp, _, sort_order = self.feature_importance()
        if n_feat is not None:
            if (n_feat < 1) or (n_feat > self.idx_unique.size):
                raise ValueError("n_feat should be an integer in the range [0,{}]. Current value: {}"
                                 .format(self.idx_unique.size, n_feat))
        elif frac_feat is not None:
            if (frac_feat <= 0) or (frac_feat > 1):
                raise ValueError("frac_feat should be an integer in the range ]0,1]. Current value: {}"
                                 .format(frac_feat))
            n_feat = int(frac_feat*self.idx_unique.size)
        elif frac_feat_imp is not None:
            if (frac_feat_imp <= 0) or (frac_feat_imp > 1):
                raise ValueError("frac_feat_imp should be an integer in the range ]0,1]. Current value: {}"
                                 .format(frac_feat_imp))
            fi_csum = np.cumsum(feat_imp)

            n_feat = int(np.sum((fi_csum/fi_csum[-1]) <= frac_feat_imp))

        bool_mask = np.full((self.idx_unique.size,), False)
        bool_mask[sort_order[:n_feat]] = True
        return bool_mask

    def set_mask(self, mask: np.ndarray = None, **kwargs):
        if mask is not None:
            if mask.dtype == bool:
                self.mask_idx = mask
            else:
                raise TypeError("mask should have dtype bool. Current type: {}".format(mask.dtype))
        else:
            self.mask_idx = self.feature_selection(**kwargs)

