# BSD 3-Clause License
#
# Copyright (c) 2025, Rikard Vinge
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from math import comb
import numpy as np
import itertools as it
from typing import List, Tuple
from sklearn.svm import SVC
from .plot import waterfall, bar


class InteractionUtils:
    """
    Feature interaction utilities.

    Helper class to identify and enumerate interactions.

    Interactions of dimension d can be viewed as elements of a rank-d tensor. The tensor is permutation symmetric,
    i.e. to elements with indices that are permutations of each other are equal. The tensors are also assumed to have
    the same number of elements in each dimension.

    For example: The values at locations (i,i,k), (i,k,i) and (k,i,i) are all assumed to be identical.

    This type of tensors occur any time a vector is multiplied by itself but along different dimensions, e.g. x*x^T.

    Parameters
    ----------
    interaction_dim : int
        Dimension of interaction.
    n_feature : int
        Number of features in original space.

    """

    def __init__(self, interaction_dim: int, n_feature: int) -> None:
        self.interaction_dim = interaction_dim
        self.n_feature = n_feature

    def create_unique_index(self) -> List[str]:
        """
        Create set of interactions where the order of the indexes does not matter.
        That is, the interaction is symmetric such that the elements at index (i,j,k,l) and
        (i,k,j,l) are identical for any number of features and interaction dimension.

        :return: List[str].

        Returns
        -------
        idx_strs : List[str]
            List of indices of unique interaction such that values at permuted interactions are equal.
        """
        list_idx = list(it.combinations_with_replacement(np.arange(self.n_feature), self.interaction_dim))
        return [','.join([str(idx) for idx in tup]) for tup in list_idx]

    @classmethod
    def _count_index_occurrences(cls, idx_list) -> Tuple[List[str], List[str]]:
        """
        Count the number of occurrences of each index in the interaction.

        For example, the element (0,0,1,2,3) has occurrences 2,1,1,1,0.
        corresponding to two 0s, one 1, one 2, one 3 and zero 4. The 4 ins included since this
        is the highest possible index in this interaction.

        Parameters
        ----------
        idx_list : List[str].
            List of strings with tensor indexes.

        Returns
        -------
        occurrences_counts : List[str]
            Count of number of occurrences for each index in idx_list.
        occurrences_counts : List[str]
            Unique elements of elem_list.
        """
        occurrences_count_unique = set()
        occurrences_counts = []
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
            # Store counts as sorted string separated with comma. Sorted with the smallest number of occurrences first.
            counts.sort()
            # Convert to string and join with separating comma.
            counts = ','.join([str(count) for count in counts])
            occurrences_count_unique.add(counts)
            occurrences_counts.append(counts)
        return occurrences_counts, list(occurrences_count_unique)

    def n_perm(self) -> Tuple[List[str], List[int]]:
        """
        List unique interactions and the number of permutations for each interaction.

        Returns
        -------
        interactions : List[str]
            List of unique interactions given permutation symmetry of the indexes.
        perm_count : List[str]
            List of the number of possible permutations for each interaction.
        """
        interactions = self.create_unique_index()
        count_strs, count_strs_unique = self._count_index_occurrences(interactions)
        # Map number of permutations to each unique index in interactions.
        perm_count_dict = {count_str: self._count_perm(count_str) for count_str in count_strs_unique}
        perm_count = list(map(perm_count_dict.get, count_strs))
        return interactions, perm_count

    def _count_perm(self, count_str: str) -> int:
        """
        Calculate number of permutations from a string formatted as 'x,y,z,...' where x is the number of occurrences
        of the first index, y is the number of occurrences of the second index, etc.

        Example:
        Take the interaction xi^2*xj*xk*xl. This can be viewed as a tensor element located at (i,i,j,k,l).
        The number of permutations, e.g. (i,i,j,k,l),
        is then 5!/(2!)=60
        5! is from the rank of the tensor. It's possible to place 5 distinct values in 5! ways.
        Now, 2 indexes were the same, meaning the total number of unique permutations is reduced.
        The reduction is 2!, from the number of possible ways we can interchange the two identical indexes.
        Rank of tensor is 5=2+1+1+1, i.e. the number of occurrences of i,j,k,l.

        Parameters
        ----------
        count_str : str
            String of occurrences of each index in an interaction.

        Returns
        -------
        n_perm : int
            Number of permutations of the indices in count_str.
        """
        counts_list = [int(ch) for ch in count_str.split(',') if int(ch) > 1]
        if len(counts_list) == 0:
            perm_reduction = 1
        else:
            perm_reduction = np.prod([np.math.factorial(count) for count in counts_list])
        n_perm = int(np.math.factorial(self.interaction_dim) / perm_reduction)
        return n_perm


def dict2array(di) -> np.ndarray:
    """
    Flatten dict to array. Dict is assumed to have structure {1:array_1, 2:array_2,...} where array_x are 2d
    arrays where rows are assumed belonging together.

    Example: The array
    {1:np.array([[1, 2, 3], [1, 2, 3]]),2:np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
    is returned as [[1, 2, 3, 1, 2, 3, 4, 6, 9], [1, 2, 3, 1, 2, 3, 4, 6, 9]]

    Parameters
    ----------
    di : dict
        Dictionary to transform

    Returns
    -------
    Array : Numpy ndarray
        Concatenated arrays in the dict. Concatenated along axis 1.
    """
    return np.concatenate([*{key: val for key, val in di.items() if val.size > 0}.values()], axis=-1)


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

        p : int
            Number of features in original space. Calculated from sv if not provided.
            Optional, use only if no scikit-learn model is provided.

        dual_coef : numpy.ndarray of shape (n_SV, 1)
            SVM dual coefficients. Same as dual_coef_ in sklearn's SVC. Calculated as dual_coef[i] = alpha[i]*y[i].
            Optional, use only if no scikit-learn model is provided.

        intercept : float
            The constant in the SVM decision function. Optional, use only if no scikit-learn model is provided.

        kernel_d : int
            Degree of the polynomial kernel. Optional, use only if no scikit-learn model is provided.

        kernel_r : float
            Independent term of the polynomial kernel. Optional, use only if no scikit-learn model is provided.

        kernel_gamma : float or {'scale', 'auto'}
            Kernel coefficient controlling the relative importance of higher-order terms.
            Optional, use only if no scikit-learn model is provided.

        svc_model : sklearn.svm.SVC
            Trained Scikit-learn SVC model. Use to simplify creation of ExPSVM object.
            Parameters are extracted automatically from the SVC object.

        transform : bool
            Set to True to transform SVM model at creation of ExPSVM object. Otherwise, a call to transform_svm()
            is required. The transformation is done using transform_svm() without any input and may become
            computationally heavy if kernel dimension and the number of features are large. Default is False.

        feature_names : List of strings
            List of feature names in the original space.

        Attributes
        ----------
        _interactions : numpy.ndarray of str of shape (n_interactions,)
            List of unique indices that are explicitly included in the linear transformations. Formatted as
            ['i,j,k,l', 'i,i,k,l',...], for example. n_interactions is the number of explicit interactions
            in the linear model.

        _perm_count : numpy.ndarray of int of shape (n_interactions,)
            List of number of permutations of each index in _interactions.

        _interaction_dims: numpy.ndarray of int of shape (n_interactions,)
            List of dimensionality of each interaction in _interactions. For example, interaction x1*x2*x3
            has dimensionality 3.

        _poly_coef : dict
            Dictionary containing constants relating to the dimensionality of the interaction,
            i.e. different constant for first order interactions compared to second order interactions. Calculated as
            (kernel_d choose d)*r^(kernel_d - d) where d is the dimensionality of the interaction.

        interaction_mask : numpy.ndarray of bool of shape (n_interactions,)
            Boolean array with True for every element that should be kept in the linear transformation.
            The mask is used to select the most important features and reduce the size of the linear model.

        linear_model : numpy.ndarray of float of shape (n_interactions, 1)
            Linear model transformed from the polynomial SVM model. Used to extract feature importance
            and calculate the decision function.

        linear_model_is_masked : bool
            Flag to propagate the information that the linear_model has already been masked and does not
            contain all interactions. Used when calculating the decision function and the feature importance
            when evaluated on individual observations.
        """

    def __init__(self, sv: np.ndarray = None, dual_coef: np.ndarray = None,
                 intercept: float = None,
                 kernel_d: int = None, kernel_r: float = None, kernel_gamma: float = None,
                 p: int = None, svc_model: SVC = None, transform: bool = False,
                 feature_names: List[str] = None
                 ) -> None:

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
            # Number of features in original space
            if p is None:
                self.p = sv.shape[1]
            else:
                self.p = p

            # SVM model
            self.sv = sv
            # SVM dual coefficients, equal to alpha_i*y_i in standard SVM formulation.
            self.dual_coef = np.reshape(dual_coef, (-1, 1))
            # SVM intercept
            self.intercept = intercept

            # Polynomial kernel parameters
            # Polynomial degree of kernel
            self.kernel_d = kernel_d
            # Constant term in kernel
            self.kernel_r = kernel_r
            # Scale in polynomial kernel
            self.kernel_gamma = kernel_gamma

        # Instantiate compressed polynomial SVM
        # List of unique interactions
        self._interactions = np.array([])
        # List of number of permutations per interaction
        self._perm_count = np.array([])
        # List of dimensionality interaction
        self._interaction_dims = np.array([])
        # Constants used in linear model, one constant per polynomial term.
        self._poly_coef = dict()
        # Boolean array mask for feature selection
        self.interaction_mask = np.array([])

        # Numpy array with linear model
        self.linear_model = np.array([])
        # Flag to ensure decision function uses masking if linear model has been masked.
        self.linear_model_is_masked: bool = False

        # Feature names
        if feature_names is not None:
            self.feature_names: List[str] = feature_names
        else:
            self.feature_names: List[str] = None

        # Transform
        if transform:
            self.transform_svm()

    def get_interactions(self, **kwargs) -> np.ndarray:
        """
        Returns numpy array of strings of interactions.
        Example: The interaction x1*x1*x2*x3 is returned as '1,1,2,3'.

        Parameters
        ----------
        kwargs : Arguments passed to get_interaction_index(). Used to reduce number of interactions.

        Returns
        -------
        Interactions : Numpy ndarray of shape (n_interactions,) of str.
            Array of strings with interactions.
        """
        return self._interactions[self.get_interaction_index(**kwargs)]

    def get_perm_count(self, **kwargs) -> np.ndarray:
        """
        Returns numpy array of number of permutations of the interactions in _interactions.
        Example: The interaction x1*x1*x2*x3 has 4!/2!=12 possible interactions.

        Parameters
        ----------
        kwargs : Arguments passed to get_interaction_index(). Used to reduce the number of interactions.

        Returns
        -------
        Permutation count : Numpy ndarray of shape (n_interactions,)
            Array of the number of permutations for each interaction in _interactions.
        """
        return self._perm_count[self.get_interaction_index(**kwargs)]

    def get_interaction_dim(self, **kwargs) -> np.ndarray:
        """
        Returns numpy array of the dimension of each interaction in _interactions.
        Example: The interaction x1*x1*x2*x3 has dimension 4.

        Parameters
        ----------
        kwargs : Arguments passed to get_interaction_index(). Used to reduce number of interactions.

        Returns
        -------
        Interaction dimensions : Numpy ndarray of shape (n_interactions,)
            Array of the dimension of the interactions in _interactions.
        """
        return self._interaction_dims[self.get_interaction_index(**kwargs)]

    def get_linear_model(self, **kwargs) -> np.ndarray:
        """
        Returns numpy array of the linear model transformed from the polynomial SVM.
        If linear_model_is_masked is False, kwargs can be supplied to include only selected
        interactions, as controlled by get_interaction_index().

        Parameters
        ----------
        kwargs : Arguments passed to get_interaction_index(). Used to reduce the number of interactions.

        Returns
        -------
        Linear model : Numpy ndarray of shape (n_interactions, 1)
            Linear model of the polynomial SVM model.
        """
        if self.linear_model_is_masked:
            return self.linear_model
        else:
            return self.linear_model[self.get_interaction_index(**kwargs), :]

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set names for the features in original space
        Parameters
        ----------
        feature_names : List of strings
            List of feature names

        """
        self.feature_names = feature_names

    def _set_transform(self) -> None:
        """
        Set _interactions, _perm_count, _interaction_dims and _poly_coef based on the number of features and the
        chosen polynomial kernel.

        Returns
        -------
        self : Instance
            Instance with _interactions, _perm_count, _interaction_dims and _poly_coef based on the number of features
            and the chosen polynomial kernel.
        """
        for d in np.arange(1, self.kernel_d + 1):
            tp = InteractionUtils(d, self.p)
            idx_tmp, n_perm_tmp = tp.n_perm()
            self._interactions = np.concatenate((self._interactions, idx_tmp))
            self._perm_count = np.concatenate((self._perm_count, n_perm_tmp))
            self._interaction_dims = np.concatenate((self._interaction_dims, d * np.ones((len(idx_tmp),))))
            self._poly_coef[d] = comb(self.kernel_d, d) * (self.kernel_r ** (self.kernel_d - d))

    def _compress_transform(self, x: np.ndarray, reduce_memory: bool = False,
                            mask: bool = False, output_dict: bool = False):
        """
        Transform observations in x in original space into the transformed space of the polynomial kernel.

        Parameters
        ----------
        x : Numpy array of shape (n_observations, n_features)
            Observation(s) to transform. Shape of ndarray is assumed to be (n_observations, n_features). n_features are
            in the original space.
        reduce_memory : Boolean
            Set to False (default) for handling all observations and transformations at once. If True, handle one
            observation at a time for reduced memory usage.
        mask : Boolean
            If True, apply interaction_mask when computing the components of the transformation.
        output_dict : Boolean.
            If True, output dict with polynomial parts as keys. If False output numpy ndarray of shape
            (n_observations, n_interactions). Default is False.

        Returns
        -------
        transformation : Numpy ndarray of shape (n_observations, n_interactions) or dict.
            The transformed features of x. Structured as dict with keys being the polynomial degree, 1, 2,..., kernel_d.
            The order is the same as given by get_interaction_index(d, mask).
        """
        transformation = dict()
        for d in np.arange(1, self.kernel_d + 1):
            # Extract list of indices in the interactions.
            d_idx = [[int(ch) for ch in idx.split(',')] for idx in self.get_interactions(d=d, mask=mask)]
            # Create rank-3 tensor of each observation with the elements of the interaction in the last dimension.
            # Find the interactions by calculating the product along the last dimension of the tensor.
            if len(d_idx) > 0:
                if reduce_memory:
                    feat_trans = np.zeros((x.shape[0], len(d_idx)))
                    for ind, v in enumerate(x):
                        feat_trans[ind, :] = np.squeeze(np.prod(np.expand_dims(v, axis=0)[:, d_idx], axis=2))
                else:
                    feat_trans = np.squeeze(np.prod(x[:, d_idx], axis=2))

                # Reshape into (1, n_interactions) if needed.
                if len(feat_trans.shape) == 1:
                    if len(d_idx) == 1:
                        feat_trans = np.reshape(feat_trans, (feat_trans.shape[0], 1))
                    else:
                        feat_trans = np.reshape(feat_trans, (1, feat_trans.shape[0]))
                transformation[d] = feat_trans
            else:
                transformation[d] = np.array([])
        # Output either as dict or numpy ndarray.
        if not output_dict:
            transformation = dict2array(transformation)
        return transformation

    def get_interaction_index(self, interaction_strs: List[str] = None,
                              d: int = None, mask: bool = False) -> np.ndarray:
        """
        Returns indices of interactions from _interactions based on a list of interaction strings and/or interaction
        dimension and/or interactions included in interaction_mask.
        If multiple of interaction_strs, d, and mask are provided, the intersection of the provided inputs are used.
        That is, if both interaction_strs and d ir
        provided, only indices of interactions of dimension d that are also in interaction_strs are returned.

        Parameters
        ----------
        interaction_strs : List[str]
            Interactions to include.
        d : int
            Dimension to extract indexes from.
        mask : Boolean
            Set to True to extract only indexes in mask of dimension d. Default False.

        Returns
        -------
        ind : Numpy ndarray of shape (n_interactions,)
            Boolean array with True at elements that satisfy all inputs.
        """

        if mask:
            if self.interaction_mask.size == self._interactions.size:
                ind = self.interaction_mask
            else:
                raise ValueError("Instance variable mask is empty.")
        else:
            ind = np.full(self._interactions.shape, True)

        if interaction_strs is not None:
            ind = ind & np.isin(self.get_interactions(), interaction_strs)

        if d is not None:
            ind = ind & (self._interaction_dims == d)
        return ind

    def transform_svm(self, reduce_memory: bool = False, mask: bool = False):
        """
        Calculate and set a compressed linear version of the polynomial SVM model.

        Parameters
        ----------
        reduce_memory : Boolean
            Set to True to reduce memory usage by looping over support vectors instead of transforming all at once.
            Default is False.
        mask : Boolean
            Set to True to apply mask to the linear model, reducing the number of elements of the model. Note that
            masking can alternatively be applied when calculating decision functions, to, for example, evaluate feature
            selection.

        Returns
        -------
        self : Instance
            Instance with linear model.
        """

        if (self._interactions.size == 0) or (self._perm_count.size == 0):
            self._set_transform()

        # Get transformation of support vectors.
        transform = self._compress_transform(x=self.sv, reduce_memory=reduce_memory, mask=mask, output_dict=True)

        # Multiply transform by dual coefficients, sum over support vectors, and scale with polynomial coefficient
        for d in transform.keys():
            if transform[d].size > 0:
                # Multiply compressed transform by number of permutations of each interaction.
                transform_tmp = np.multiply(self.get_perm_count(d=d, mask=mask), transform[d])
                # Multiply by the SVM dual coefficients.
                transform_tmp = np.multiply(self.dual_coef, transform_tmp)
                # Sum over support vectors.
                transform_tmp = np.sum(transform_tmp, axis=0, keepdims=False)
                # Multiply by binomial coefficient and gamma constant.
                transform[d] = self._poly_coef[d] * (self.kernel_gamma ** d) * transform_tmp

        # Create linear model
        self.linear_model = np.expand_dims(dict2array(transform), axis=1)

        # Set linear_model_is_masked flag.
        if mask:
            self.linear_model_is_masked = True

    def decision_function_components(self, x: np.ndarray,
                                     include_intercept: bool = True,
                                     output_interaction_names: bool = False,
                                     reduce_memory: bool = False, mask: bool = False,
                                     format_interaction_names: bool = False):
        """
        Returns the components of the decision function for the observation(s) x.

        Parameters
        ----------
        x : Numpy ndarray of shape (n_observations, n_features)
            Observations to calculate decision function for.
        include_intercept : Boolean
            Set to True (default) to prepend the intercept to both the decision function component values and the
            interaction names.
        output_interaction_names : Boolean
            If True, return also the names of the interactions.
        reduce_memory : Boolean
            Set to True to reduce memory requirements when calculating the compressed linear transform of x.
            Default is False.
        mask : Boolean
            Set to True to apply interaction_mask to the decision function, removing some interactions.
            Default is False.
        format_interaction_names : Boolean
            Set to True to format the interaction names using the feature names, if they exist. Default is False.

        Returns
        -------
        df_comp : Numpy ndarray of shape (n_observations, n_interactions+1)
            The components of the decision function. df_comp[0] correspond to the independent component, remaining are
            given in the same order as _interactions.
        """
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(x.shape) > 2:
            raise ValueError("x should be 2-dimensional. Shape of x is {}.".format(x.shape))
        use_mask = mask or self.linear_model_is_masked
        # Transform observations to compressed linear form, possibly masked.
        x_trans = self._compress_transform(x=x, reduce_memory=reduce_memory,
                                           mask=use_mask, output_dict=False)

        # Compute the dependent components of the decision functions. These are the element-wise multiplications
        # of the observations and the linear model.
        df_comp = np.multiply(x_trans, np.transpose(self.get_linear_model(mask=use_mask)))
        # Prepend the independent component, the SVM intercept.
        if include_intercept:
            df_comp = np.concatenate((self.intercept * np.ones((x.shape[0], 1)), df_comp), axis=1)
        if output_interaction_names:
            feat_names = self.get_interactions(mask=use_mask)
            if include_intercept:
                feat_names = np.append(['intercept'], feat_names)
            if format_interaction_names:
                feat_names = self.format_interaction_names(feat_names)
            return df_comp, feat_names
        else:
            return df_comp

    def decision_function(self, x: np.ndarray, reduce_memory: bool = False, mask: bool = False):
        """
        Returns the decision function value for observation(s) x.

        Parameters
        ----------
        x : Numpy ndarray of shape (n_observations, n_features)
            Observations to calculate decision function for.
        reduce_memory : Boolean
            Set to True to reduce memory requirements when calculating the compressed linear transform of x.
            Default is False.
        mask : Boolean
            Set to True to apply interaction_mask to the decision function, removing some interactions.
            Default is False.

        Returns
        -------
        X : Numpy ndarray of shape (n_observations,)
            Decision function for each observation.
        """
        dot_prod = self.decision_function_components(x=x, output_interaction_names=False,
                                                     reduce_memory=reduce_memory, mask=mask)
        return np.sum(dot_prod, axis=1, keepdims=False)

    def feature_importance(self, sort: bool = True, format_names: bool = False, magnitude: bool = True,
                           include_intercept: bool = True, **kwargs):
        """
        Calculate feature importance and return feature importance, feature names and sorting order.

        Parameters
        ----------
        sort : Boolean
            If True (default) sort the features in order of importance.
        format_names : Boolean
            If True, format interaction strings using format_interaction_names().
        magnitude : Boolean
            If True (default) return the absolute value of the feature importance, otherwise return the signed
            importance.
        include_intercept : Boolean
            If True (default) append intercept to list of feature importance.
        kwargs : Arguments passed to get_linear_model.

        Returns
        -------
        feat_importance : Numpy ndarray of shape (n_interactions,)
            Array with feature importance.
        feat_names : Numpy ndarray of shape (n_interactions,)
            Names of the features.
        sort_order : Numpy ndarray of shape (n_interactions,)
            Sorting order to get feat_names from _interactions.
        """

        # Get interaction importance
        feat_importance_signed = np.squeeze(self.get_linear_model(**kwargs)[:, 0])

        # Get feature names
        feat_names = self.get_interactions(**kwargs)

        # Append intercept
        if include_intercept:
            feat_importance_signed = np.append([self.intercept], feat_importance_signed)
            feat_names = np.append(['intercept'], feat_names)

        # Calculate magnitudes of interaction importance
        feat_importance = np.abs(feat_importance_signed)

        # Order feature importance and names
        if sort:
            sort_order = np.argsort(feat_importance)[::-1]
        else:
            sort_order = np.arange(feat_importance.size)
        feat_importance = feat_importance[sort_order]
        feat_importance_signed = feat_importance_signed[sort_order]
        feat_names = feat_names[sort_order]

        if format_names:
            feat_names = self.format_interaction_names(feat_names)

        if magnitude:
            return feat_importance, feat_names, sort_order
        else:
            return feat_importance_signed, feat_names, sort_order

    def feature_selection(self, n_interactions: int = None,
                          frac_interactions: float = None,
                          frac_importance: float = None):
        """
        Return the most important features given either: a set number of interactions; a fraction of the interactions;
        or a fraction of the interaction importance.

        Importance is measured by the magnitude of the interaction in the linear model.

        Parameters
        ----------
        n_interactions : int
            Number of the most important interactions to select.
        frac_interactions : float in range [0,1]
            Fraction of the most important interactions to select.
        frac_importance : float in range [0,1]
            Select number of interactions given by the total contribution to the decision function.

        Returns
        -------
        interaction_mask : Boolean numpy ndarray of shape (n_interactions,)
            Boolean array where True at elements that are judges as important by the input constraints.
        """

        feat_imp, _, sort_order = self.feature_importance(include_intercept=False)

        if n_interactions is not None:
            if (n_interactions < 1) or (n_interactions > self._interactions.size):
                raise ValueError("n_feat should be an integer in the range [0,{}]. Current value: {}"
                                 .format(self._interactions.size, n_interactions))
        elif frac_interactions is not None:
            if (frac_interactions <= 0) or (frac_interactions > 1):
                raise ValueError("frac_feat should be an integer in the range ]0,1]. Current value: {}"
                                 .format(frac_interactions))
            n_interactions = int(frac_interactions * self._interactions.size)
        elif frac_importance is not None:
            if (frac_importance <= 0) or (frac_importance > 1):
                raise ValueError("frac_feat_imp should be an integer in the range ]0,1]. Current value: {}"
                                 .format(frac_importance))
            fi_csum = np.cumsum(feat_imp)

            n_interactions = int(np.sum((fi_csum / fi_csum[-1]) <= frac_importance))
        else:
            n_interactions = self._interactions.size

        interaction_mask = np.full((self._interactions.size,), False)
        interaction_mask[sort_order[0:n_interactions]] = True
        return interaction_mask

    def set_mask(self, mask: np.ndarray = None, interaction_strs: List[str] = None, **kwargs):
        """
        Set mask for the compressed linear transformation of the polynomial SVM.

        Parameters
        ----------
        mask : Boolean Numpy ndarray of shape (n_interactions,).
            Interactions to include.
        interaction_strs : List[str]
            Interactions to include in the mask. Interactions should have format: 'i,j,k,...', e.g. '0,0'.
        kwargs : Arguments passed to feature_selection().

        Returns
        -------
        self : Instance.
            Instance with interaction_mask.
        """
        if mask is not None:
            if mask.dtype == bool:
                self.interaction_mask = mask
            else:
                raise TypeError("mask should have dtype bool. Current type: {}".format(mask.dtype))
        elif interaction_strs:
            if interaction_strs is not None:
                self.interaction_mask = self.get_interaction_index(interaction_strs=interaction_strs)
        else:
            self.interaction_mask = self.feature_selection(**kwargs)

    def format_interaction_names(self, interaction_strs: List[str]) -> List[str]:
        """
        Return formatted interaction strings from 'i,i,j,k' to 'x_{i}^{2}x_{j}x_{k}'. Excludes unused features.

        Example:
        The interaction_strs = ['0,0,1', '0,1,2', '0,1,0,2'] is returned as ['x_{0}^{2}x_{1}', 'x_{0}x_{1}x_{2}',
        'x_{0}^{2}x_{1}x_2'].

        Parameters
        ----------
        interaction_strs : List[str]
            List of interactions strings formatted as returned by InteractionUtils.create_unique_index().

        Returns
        -------
        formatted_strs : List[str]
            List of formatted interaction strings.
        """
        formatted_strs = []
        if self.feature_names is None:
            interactions = [[int(ind) if ind != 'intercept' else 'intercept'
                             for ind in ind_str.split(',')] for ind_str in interaction_strs]
            for interaction in interactions:
                indices = np.arange(self.p)
                counts = [np.count_nonzero(interaction == ind) for ind in indices]
                counts_str = ''.join(
                    ['$x_{{{}}}^{{{}}}$'.format(i, c) if c > 1 else '$x_{{{}}}$'.format(i) if c == 1 else ''
                     for i, c in enumerate(counts)])

                formatted_strs.append(counts_str)
        else:
            f_name_dict = {str(num): name for num, name in enumerate(self.feature_names)}
            f_name_dict['intercept'] = 'intercept'  # Add constant to feature name dictionary
            for ind, feat_name in enumerate(interaction_strs):
                interaction_name = ')*('.join([f_name_dict[f] for f in feat_name.split(',')])
                if '*' not in interaction_name:
                    interaction_name = interaction_name.replace('(', '').replace(')', '')
                else:
                    interaction_name = '(' + interaction_name + ')'
                formatted_strs.append(interaction_name)
        return formatted_strs

    def plot_model_bar(self, n_features: int = 10, magnitude: bool = False,
                       include_intercept: bool = False, **kwargs):
        """
        Visualize the weights of the interactions in the decision function of 
        an SVM trained with polynomial kernel using a bar chart. Interactions
        are ordered from the highest weight to the lowest.

        Parameters
        ----------
        n_features : Int
            Number of features to show. Remaining features will be ignored
        magnitude : Boolean
            Set to True to visualize the magnitude of the importance. If False
            the weight will be shown with it's corresponding sign. Default is False.
        include_intercept : Boolean
            Set to True to include the intercept among the weights. Default is False.
        figsize : Tuple of two integers
            Size of the pyplot graph. Should be of the format [w, h] or (w, h) where w and h are integers. 
        kwargs : Comma-separated list key-value pairs
            Arguments to forward to plot._bar.bar.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        feat_importance, feat_names, _ = self.feature_importance(format_names=True,
                                                                          magnitude=magnitude,
                                                                          include_intercept=include_intercept)

        # If number of features to plot is too large, set it to number of existing features.
        if len(feat_importance) < n_features:
            n_features = len(feat_importance)

        # Get bar heights and labels
        bar_heights = feat_importance[0:n_features]
        labels = feat_names[0:n_features]

        xlabel = 'Interaction'
        ylabel = 'Decision function weight'
        title = f'Top {n_features} most important interactions'

        if magnitude:
            ylabel += ' magnitude'
            title += ' (magnitude)'
        return bar(bar_heights, labels, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)

    def plot_sample_waterfall(self, x: np.ndarray, n_features: int = 10, **kwargs):
        """
        Visualize interaction importance for a single observation using a waterfall graph.

        Parameters
        ----------
        x : np.ndarray
            Sample to visualize. Should have shape (n_original_features,).
        n_features : int
            Number of features to explicitly show. Any remaining features will be bunched together into a "Remaining"
            bar. Default is 10. n_features is set to n_original_features if n_original_features < n_features.
        kwargs : Comma-separated list key-value pairs
            Arguments to forward to plot._waterfall.waterfall

        Returns
        -------
        matplotlib.figure.Figure or None
        """

        # Check validity of observation
        x = np.squeeze(x)
        if len(x.shape) > 1:
            ValueError(
                'Input observation x should have dimension (n_feature,). Shape of provided input {}'.format(x.shape))

        # Calculate decision function value and components
        y_comp, feat_names = self.decision_function_components(x=x, include_intercept=False,
                                                               output_interaction_names=True,
                                                               format_interaction_names=True)
        y_comp = y_comp[0]

        # Reset number of features to show if value is too high.
        if y_comp.size < n_features:
            n_features = y_comp.size

        # Get order of interactions in descending order of importance
        sort_order = np.flip(np.argsort(np.abs(y_comp)))

        # Reorder components and interaction names
        y_comp_sort = y_comp[sort_order]
        feat_names_sort = list(np.array(feat_names)[sort_order])

        # Instantiate array of bar widths and labels
        bar_widths = np.concatenate(([self.intercept], y_comp_sort[0:n_features]))
        labels = ['Intercept'] + feat_names_sort[0:n_features]
        # labels = np.concatenate((['intercept'], feat_names_sort[0:n_features]))

        # Check if there are any interactions that are not shown
        n_remaining = y_comp_sort.size - n_features
        if n_remaining > 0:
            bar_widths = np.append(bar_widths, y_comp_sort[n_features:].sum())
            labels.append(f'Remaining {n_remaining} interactions')

        return waterfall(bar_widths, labels, **kwargs)

    def plot_sample_waterfall_degree(self, x: np.ndarray, n_degree: int = None, **kwargs):
        # Check validity of observation
        x = np.squeeze(x)
        if len(x.shape) > 1:
            ValueError(
                'Input observation x should have dimension (n_feature,). Shape of provided input {}'.format(x.shape))

        if (n_degree is None) | (n_degree > self.kernel_d):
            n_degree = self.kernel_d

        # Calculate decision function value and components
        y_comp, feat_names = self.decision_function_components(x=x, include_intercept=False,
                                                               output_interaction_names=True,
                                                               format_interaction_names=True)

        bar_widths = np.array([self.intercept] + [y_comp[0, self._interaction_dims == d].sum()
                                                  for d in np.arange(1, n_degree+1)])
        labels = ['Intercept'] + [f'Degree {i}' for i in np.arange(1, n_degree+1)]

        # Check if there are any interactions that are not shown
        n_remaining = self.kernel_d - n_degree
        if n_remaining > 0:
            bar_widths = np.append(bar_widths, y_comp[0, self._interaction_dims > n_degree].sum())
            # labels = np.append(labels, f'Remaining {n_remaining} interactions')
            labels.append(f'Remaining {n_remaining} degrees')

        return waterfall(bar_widths, labels, **kwargs)




