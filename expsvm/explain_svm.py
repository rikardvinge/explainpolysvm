from math import comb
import numpy as np
import itertools as it
from typing import List
from sklearn.svm import SVC


class TensorUtil:
    def __init__(self, rank: int, dim: int) -> None:
        self.rank = rank
        self.dim = dim

    def create_unique_index(self) -> List[str]:
        """
        Create set of tensor indexes where the order of the indexes does not matter.
        That is, the tensor is symmetric such that the elements at index (i,j,k,l) and
        (i,k,j,l) are identical for any number of indices and tensor rank.

        :return: List[str]. List of strings containing indices formatted as 'i,j,k,l,...'
        where i,j,k,l,...= 1..p where p is the dimension of the tensor and the number of
        indices i,j,k,l is equal to the tensor dimension. Note that multiple of i,j,k,l,...
        can take the same values.
        """
        list_idx = list(it.combinations_with_replacement(np.arange(self.dim), self.rank))
        return [','.join([str(idx) for idx in tup]) for tup in list_idx]

    @classmethod
    def _count_index_occurrences(cls, idx_list):
        """
        Count the number of occurrences of each index.
         For example, the element (0,0,1,2,3) has occurrences 2,1,1,1,0.
        corresponding to two 0s, one 1, one 2, one 3 and zero 4. The 4 ins included since this
        is the highest possible index in this tensor.

        :param idx_list: List[str]. List of strings with tensor indexes.
        :return elem_list: List[int]. Count of number of permutations for each index in idx_list.
        :return counts_unique: List[int]. Unique elements of elem_list.
        """
        counts_unique = set()
        elem_counts = []
        for elem in idx_list:
            # Convert string to list of int.
            # This is to ensure we count, for example, '1,11' as one 1 and one 11, not three 1s and one 11.
            idx_list = [int(idx) for idx in elem.split(',')]
            idx_set = set(idx_list)

            # Count occurrences of each index
            counts = []
            for idx in idx_set:
                count = idx_list.count(idx)
                counts.append(count)
            # Store counts as sorted string separated with comma. Sorted with lowest number of occurrences first.
            counts.sort()
            counts = ','.join([str(sorted_count) for sorted_count in counts])
            counts_unique.add(counts)
            elem_counts.append(counts)
        return elem_counts, list(counts_unique)

    def n_perm(self) -> (List[str], List[int]):
        """
        List unique tensor indices and the number of permutations for each index.
        :return idx_unique: List[str]. List of unique tensor indices given permutation symmetry of the indexes.
        :return idx_n_perm: List[int]. List of counts of permutation for each index in idx_unique.
        """
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
        :return n_perm: int. Number of possible permutations
        """
        counts_list = [int(ch) for ch in count_str.split(',') if int(ch) > 1]
        if len(counts_list) == 0:
            perm_reduction = 1
        else:
            perm_reduction = np.prod([np.math.factorial(count) for count in counts_list])
        n_perm = int(np.math.factorial(self.rank) / perm_reduction)
        return n_perm


class ExPSVM:
    """
        Explain polynomial SVM.

        Transformed a trained SVM model into a linear model where each term in the model corresponds
        to an interaction between the original features. The model assumes the SVM is trained with
        a polynomial kernel defined as
        K(x,y|d,r,gamma) = (r + gamma*(x^Ty))^d.
        Here, r is the constant term of the polynomial kernel, d the polynomial degree, gamma
        a kernel coefficient determining the relative weight of higher components, and x and y feature vectors.

        The transformation to a linear kernel makes use of the fact that interactions containing the exact same
        constituents, all have the same value. For example, the interaction x1*x1*x3*x5, that is, feature 1 multiplied
        by itself as well as feature 3 and 5, is equal to X1*x3*x5*x1. Thus, we only need to calculate one of these
        interactions. In the linear model, the total contribution from this interaction is calculated from the
        value of one of the interaction multiplied by the number of identical interactions.

        Parameters
        ----------
        sv : numpy.ndarray of shape (n_SV, n_features)
            Support vectors. Optional, use only if no scikit-learn model is provided.

        p: int
            Number of features in original space. Calculated from sv.shape if not provided.
            Optional, use only if no scikit-learn model is provided.

        dual_coef: numpy.ndarray of shape (n_SV, 1)
            SVM dual coefficients. Same as dual_coef_ in sklearn's SVC. Calculated as dual_coef[i] = alpha[i]*y[i].
            Optional, use only if no scikit-learn model is provided.

        intercept: float
            The constant in the SVM decision function. Optional, use only if no scikit-learn model is provided.

        kernel_d: int
            Degree of the polynomial kernel. Optional, use only if no scikit-learn model is provided.

        kernel_r: float
            Independent term of the polynomial kernel. Optional, use only if no scikit-learn model is provided.

        kernel_gamma: float or {'scale', 'auto'}
            Kernel coefficient controlling the relative importance of higher-order terms.
            Optional, use only if no scikit-learn model is provided.

        svc_model: sklearn.svm.SVC
            Trained Scikit-learn SVC model. Use to simplify creation of ExPSVM object.
            Parameters are extracted automatically from the SVC object.

        Attributes
        ----------

        idx_unique: numpy.ndarray of str of shape (n_transform,)
            List of unique indices that are explicitly included in the linear tranformations. Formatted as
            ['i,j,k,l', 'i,i,k,l',...], for example. n_transform is the number of explicit interactions
            in the linear model.

        perm_count: numpy.ndarray of int of shape (n_transform,)
            List of number of permutations of each index in idx_unique.

        idx_dim: numpy.ndarray of int of shape (n_transform,)
            List of dimensionality of each interaction in idx_unique. Fo example, interaction x1*x2*x3
            has dimensionality 3.

        poly_coef: dict
            Dict containing constants relating to the dimensionality of the interaction, i.e. different constant for
            first order interactions compared to second order interactions.
            Calculated as (kernel_d choose d)*r^(kernel_d - d) where d is the dimensionality of the interaction.

        mask_idx: numpy.ndarray of bool of shape (n_transform,)
            Boolean array with True for every element that should be kept in the linear transformation.
            The mask is used to select the most important features and reduce the size of the linear model.

        linear_model: numpy.ndarray of float of shape (n_transform, 1)
            Linear model transformed from the polynomial SVM model. Used to extract feature importance
            and calculate the decision function.

        linear_model_is_masked: bool
            Flag to propagate the information that the linear_model has already been masked and does not
            contain all interactions. Used when calculating the decision function and the feature importance
            when evaluated on individual observations.
        """

    def __init__(self, sv: np.ndarray = None, dual_coef: np.ndarray = None,
                 intercept: float = None,
                 kernel_d: int = None, kernel_r: float = None, kernel_gamma: float = None,
                 p: int = None, svc_model: SVC = None) -> None:

        if svc_model is not None:
            if svc_model.classes_.size != 2:
                raise ValueError("Number of classes should be 2. "
                                 "Current number of classes: {}.".format(svc_model.classes_.size))
            # Support vectors
            self.sv = svc_model.support_vectors_
            # Number of features in original space
            self.p = svc_model.n_features_in_
            # SVM dual coefficients, equal to alpha_i*y_i in standard SVM formulation.
            self.dual_coef = np.reshape(svc_model.dual_coef_, (-1, 1))
            # SVM intercept
            self.intercept = svc_model.intercept_[0]
            # Polynomial degree of kernel
            self.kernel_d = svc_model.degree
            # Constant term in kernel
            self.kernel_r = svc_model.coef0
            # Scale in polynomial kernel
            self.kernel_gamma = svc_model._gamma
        else:
            # Number of features
            if p is None:
                self.p = sv.shape[1]
            else:
                self.p = p

            # SVM model
            self.sv = sv
            self.dual_coef = np.reshape(dual_coef, (-1, 1))
            self.intercept = intercept

            # Polynomial kernel parameters
            self.kernel_d = kernel_d
            self.kernel_r = kernel_r
            self.kernel_gamma = kernel_gamma

        # Instantiate compressed polynomial SVM
        self.idx_unique = np.array([])
        self.perm_count = np.array([])
        self.idx_dim = np.array([])
        self.poly_coef = {d: comb(self.kernel_d, d) * (self.kernel_r ** (self.kernel_d - d))
                          for d in np.arange(1, self.kernel_d + 1)}
        self.mask_idx = np.array([])

        self.linear_model = np.array([])
        self.linear_model_is_masked: bool = False

    def get_idx_unique(self, **kwargs) -> np.ndarray:
        return self.idx_unique[self.get_dim_mask_index(**kwargs)]

    def get_perm_count(self, **kwargs) -> np.ndarray:
        return self.perm_count[self.get_dim_mask_index(**kwargs)]

    def get_idx_dim(self, **kwargs) -> np.ndarray:
        return self.idx_dim[self.get_dim_mask_index(**kwargs)]

    def get_linear_model(self, **kwargs) -> np.ndarray:
        if self.linear_model_is_masked:
            return self.linear_model
        else:
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

    def get_dim_mask_index(self, idx_strs: List[str] = None, d: int = None, mask: bool = False) -> np.ndarray:
        """
        Returns index in transformed space of dimension d that are in mask.

        :param idx_strs:
        :param d: Dimension to extract indexes from.
        :param mask: Set to True to extract only indexes in mask of dimension d. Default False.
        :return:
        """

        if mask:
            if self.mask_idx.size > 0:
                ind = self.mask_idx
            else:
                raise ValueError("Instance variable mask is empty.")
        else:
            ind = np.full(self.idx_unique.shape, True)

        if idx_strs is not None:
            ind = ind & np.isin(self.get_idx_unique(), idx_strs)

        if d is not None:
            ind = ind & (self.idx_dim == d)

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
                compressed_transform[d] = self.poly_coef[d] * (self.kernel_gamma**d) * (
                    np.sum(np.multiply(self.dual_coef,
                                       np.multiply(self.get_perm_count(d=d, mask=mask), compressed_transform[d])),
                           axis=0, keepdims=False))

        # Create linear model
        self.linear_model = np.expand_dims(self.dict2array(compressed_transform), axis=1)
        if mask:
            self.linear_model_is_masked = True

    def polynomial_kernel(self, x: np.ndarray, y: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(x.shape) > 2:
            raise ValueError("x should be 2-dimensional. Shape of x is {}".format(x.shape))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))
        if len(y.shape) > 2:
            raise ValueError("y should be 2-dimensional. Shape of y is {}".format(y.shape))
        return (self.kernel_r + np.sum(np.multiply(x, y), axis=1, keepdims=False)) ** self.kernel_d

    def decision_function_components(self, x: np.ndarray, output_feat_names: bool = False,
                                     reduce_memory: bool = False, mask: bool = False):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(x.shape) > 2:
            raise ValueError("x should be 2-dimensional. Shape of x is {}.".format(x.shape))
        use_mask = mask or self.linear_model_is_masked
        x_trans = self.dict2array(self._compress_transform(x=x, reduce_memory=reduce_memory, mask=use_mask))

        dot_prod = np.multiply(x_trans, np.transpose(self.get_linear_model()))
        constant = self.intercept

        dot_prod = np.concatenate((constant * np.ones((x.shape[0], 1)), dot_prod), axis=1)
        if output_feat_names:
            feat = np.concatenate((['constant'], self.get_idx_unique(mask=use_mask)), axis=0)
            return dot_prod, feat
        else:
            return dot_prod

    def decision_function(self, x: np.ndarray, reduce_memory: bool = False, mask: bool = False):
        dot_prod = self.decision_function_components(x=x, output_feat_names=False,
                                                     reduce_memory=reduce_memory, mask=mask)
        return np.sum(dot_prod, axis=1, keepdims=False)

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
                          frac_importance: float = None):
        """

        :param n_feat:
        :param frac_feat:
        :param frac_importance:
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
            n_feat = int(frac_feat * self.idx_unique.size)
        elif frac_importance is not None:
            if (frac_importance <= 0) or (frac_importance > 1):
                raise ValueError("frac_feat_imp should be an integer in the range ]0,1]. Current value: {}"
                                 .format(frac_importance))
            fi_csum = np.cumsum(feat_imp)

            n_feat = int(np.sum((fi_csum / fi_csum[-1]) <= frac_importance))

        bool_mask = np.full((self.idx_unique.size,), False)
        bool_mask[sort_order[:n_feat]] = True
        return bool_mask

    def set_mask(self, mask: np.ndarray = None, mask_strs: List[str] = None, **kwargs):
        if mask is not None:
            if mask.dtype == bool:
                self.mask_idx = mask
            else:
                raise TypeError("mask should have dtype bool. Current type: {}".format(mask.dtype))
        elif mask_strs:
            if mask_strs is not None:
                self.mask_idx = self.get_dim_mask_index(idx_strs=mask_strs)
        else:
            self.mask_idx = self.feature_selection(**kwargs)
