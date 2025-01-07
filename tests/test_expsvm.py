import pytest
import numpy as np
from .context import expsvm as exp
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from typing import Tuple
import matplotlib

# Fix so pytest assert with numpy datatypes work with numpy >= 2.
if np.__version__ >= "2.0.0":
    import math
    np.math = math

# Create random number generator for repeatable tests.
rng = np.random.default_rng(101)

@pytest.fixture
def std_p():
    return 3


@pytest.fixture
def std_d():
    return 2


@pytest.fixture
def std_r():
    return 1.


@pytest.fixture
def std_gamma():
    return 1.


@pytest.fixture
def std_intercept():
    return 2.


@pytest.fixture
def std_dual_coef():
    return np.array([[10], [-0.1]])


@pytest.fixture
def std_arr():
    return np.array([[1., 2., 3.],
                     [4., 5., 6.]])


@pytest.fixture
def std_idx():
    return np.array(['0', '1', '2',
                     '0,0', '0,1', '0,2', '1,1', '1,2', '2,2'])


@pytest.fixture
def std_perm_count():
    return np.array([1, 1, 1,
                     1, 2, 2, 1, 2, 1])


@pytest.fixture
def std_dim():
    return np.array([1, 1, 1,
                     2, 2, 2, 2, 2, 2])


@pytest.fixture
def std_transf_dict():
    return {1: np.array([[1., 2., 3.],
                         [4., 5., 6.]]),
            2: np.array([[1., 2., 3., 4., 6., 9.],
                         [16., 20., 24., 25., 30., 36.]])}


@pytest.fixture
def std_lin_model():
    # When using std_dual_coef with std_arr
    return np.transpose(np.array([[19.2, 39, 58.8,
                                   8.4, 36, 55.2, 37.5, 114, 86.4]]))


@pytest.fixture
def std_mask():
    return np.array([False, False, False,
                     True, False, True, False, False, False])


def polynomial_kernel(x: np.ndarray, y: np.ndarray, r: float, d: int, gamma: float):
    """
    Calculate standard polynomial kernel K(x,y|r,d,gamma) = (r + gamma*(x^Ty))^d.

    Parameters
    ----------
    x : Numpy ndarray of shape (n_observations, n_feature)
        Observations
    y : Numpy ndarray of shape (n_observations, n_feature)
        Observations
    r : float
        Kernel independent coefficient.
    d : int
        Polynomial degree of kernel.
    gamma : float
        Kernel coefficient.

    Returns
    -------
    K : Numpy ndarray of shape (n_observations,) containing the Gram matrix of x and y.
        Gram matrix with the polynomial kernel.
    """
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    if len(x.shape) > 2:
        raise ValueError("x should be 2-dimensional. Shape of x is {}".format(x.shape))
    if len(y.shape) == 1:
        y = y.reshape((1, -1))
    if len(y.shape) > 2:
        raise ValueError("y should be 2-dimensional. Shape of y is {}".format(y.shape))
    return (r + gamma * np.matmul(x, np.transpose(y))) ** d


def create_sklearn_expsvm_models(X_train, y_train, C, degree, gamma, r) -> Tuple[SVC, exp.ExPSVM]:
    """
    Train a Scikit-learn SVC svm_model with a polynomial kernel and transform it using ExpSVM. Return both models.

    Parameters
    ----------
    X_train : Numpy ndarray of shape (n_observations, n_features)
        Training data
    y_train : Numpy ndarray of shape (n_observations,)
        Training labels
    C : float
        Regularization parameter in SVM
    degree : int
        Degree of the polynomial kernel
    gamma : float or {'scale', 'auto'}
        Coefficient in the polynomial kernel
    r : float
        Independent term in polynomial kernel

    Returns
    -------
    svm_model : sklearn.svm.SVC
        Trained Scikit-learn SVC model with polynomial kernel.
    es_model : ExPSVM
        Polynomial SVM model transformed into a compressed linear model
    """
    # Fit SVM
    kernel = 'poly'
    svm_model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=r)
    svm_model.fit(X_train, y_train)

    es_model = exp.ExPSVM(svc_model=svm_model)
    es_model.transform_svm()
    return svm_model, es_model


class TestExPSVM:
    def test_create_index(self, std_d, std_p, std_idx):
        """
        Verify that the TensorUtil.create_unique_index() produce a correct set of tensor indices.
        """
        tp = exp.InteractionUtils(std_d, std_p)
        test_set = set(tp.create_unique_index())
        true_set = set(std_idx[3:])
        sym_intersect = true_set.symmetric_difference(test_set)
        assert len(sym_intersect) == 0

    def test_count_occurrences(self, std_d, std_p):
        """
        Verify that TensorUtil._count_index_occurrences() counts the correct number of occurrences of each index.
        For example, elements (i,i) have count '2' while (i,j) has count 1,1.
        """
        tp = exp.InteractionUtils(std_d, std_p)
        idx_list = tp.create_unique_index()
        idx_count, unique_idx_count = tp._count_index_occurrences(idx_list)
        compare = []
        for ind, idx in enumerate(idx_list):
            idx = idx.split(',')
            if idx[0] == idx[1]:
                compare.append(idx_count[ind] == '2')
            else:
                compare.append(idx_count[ind] == '1,1')
        assert all(compare)

    def test_count_perm_3d(self):
        """
        Check that the number of permutations in 3d tensor.
        The element used is of the kind (i,i,j).
        This kind of element has 3 permutations (i,i,j), (i,j,i) and (j,i,i)
        """
        p = 4
        d = 3
        tp = exp.InteractionUtils(d, p)
        count_str = '1,2'
        n_perm = tp._count_perm(count_str)
        assert n_perm == 3

    def test_count_perm_5d(self):
        """
        Check that the number of permutations in 5d tensor.
        The element used is of the kind (i,j,j,k,k).
        This kind of element have 30 possible permutations:
        i first: ijjkk, ikjkj, ikkjj, ijkkj - 4 permutations
        i 2nd: jijkk, jikjk, jikkj, kijjk, kijkj, kikjj - 6 permutations.
        i 3rd: jjikk, jikkj, jkijk, kjikj, kjijk, kkijj - 6 permutations
        i 4th: jjkik, jkjik, jkkij, kjjik, kjkij, kkjij - 6 permutations
        i 5th: jjkki, jkjki, jkkji, kjkji, kjjki, kkjji - 6 permutations
        Total: 30
        """
        p = 8
        d = 5
        tp = exp.InteractionUtils(d, p)
        count_str = '1,2,2'
        n_perm = tp._count_perm(count_str)
        assert n_perm == 30

    def test_n_perm(self, std_d, std_p, std_idx, std_perm_count):
        """
        Verify that TensorUtil.n_perm() finds the correct tensor indices and number of permutations of each index.
        """
        tu = exp.InteractionUtils(std_d, std_p)
        idx_list, n_perm = tu.n_perm()

        # Check that all indexes exist
        assert set(idx_list) == set(std_idx[3:])

        tu_order = np.argsort(idx_list)
        idx_order = np.argsort(std_idx[3:])
        assert np.all(np.array(n_perm)[tu_order] == std_perm_count[3:][idx_order])

    def test_multiplication_transform(self, std_d, std_p, std_r,
                                      std_idx, std_perm_count, std_dim,
                                      std_gamma):
        """
        Verify that ExPSVM produce the correct interactions, permutation counts and interaction dimensions.
        """
        es = exp.ExPSVM(sv=None, dual_coef=np.array([0]),
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)
        es._set_transform()

        # Check that all indexes exist
        assert set(es._interactions) == set(std_idx)

        # Check that permutation counts and dimensions are correct
        # But first, order index
        es_order = np.argsort(es._interactions)
        idx_order = np.argsort(std_idx)
        assert np.all(es._perm_count[es_order] == std_perm_count[idx_order])
        assert np.all(es._interaction_dims[es_order] == std_dim[idx_order])

    def test_dict2array(self, std_transf_dict):
        """
        Verify that dict2array concatenates each dict value along the rows.
        """
        # Test with two samples
        arr1 = exp.dict2array(std_transf_dict)

        # Test with single observation with format (1,p)
        di2 = {1: std_transf_dict[1][0:1, :],
               2: std_transf_dict[2][0:1, :]}
        arr2 = exp.dict2array(di2)

        # Test with single observation with format (p,).
        di3 = {1: std_transf_dict[1][0, :],
               2: std_transf_dict[2][0, :]}
        arr3 = exp.dict2array(di3)
        assert np.all(arr1 == [[1., 2., 3., 1., 2., 3., 4., 6., 9.],
                               [4, 5, 6, 16., 20., 24., 25., 30., 36.]])
        assert arr1.shape == (2, 9)
        assert np.all(arr2 == [[1., 2., 3., 1., 2., 3., 4., 6., 9.]])
        assert arr2.shape == (1, 9)
        assert np.all(arr3 == [1., 2., 3., 1., 2., 3., 4., 6., 9.])
        assert arr3.shape == (9,)

    def test_interaction_index(self, std_p, std_d, std_r, std_mask, std_dim, std_gamma):
        """
        Verify that ExPSVM.get_interaction_index() returns the correct lists indices of interactions from
        ExPSVM._interactions, but all indices and with specified dimensions and/or mask.
        """
        es = exp.ExPSVM(sv=None, dual_coef=np.array([0]),
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)
        es._set_transform()
        es.interaction_mask = std_mask

        # Test get all indices
        ind_full = es.get_interaction_index()
        assert np.all(ind_full == np.full(es._interactions.shape, True))

        # Test get dimension indices only
        ind_d = es.get_interaction_index(d=std_d)
        assert np.all(ind_d == (std_dim == 2))

        # Test get mask indices only
        ind_m = es.get_interaction_index(mask=True)
        assert np.all(ind_m == std_mask)

        # Test get dimension and mask indices
        es.interaction_mask = np.array([True, False, False, True, False, True, False, False, False])
        ind_d_m = es.get_interaction_index(d=2, mask=True)
        assert np.all(ind_d_m == std_mask)

    def test_compressed_transform(self, std_p, std_d, std_r, std_gamma,
                                  std_transf_dict, std_arr,
                                  reduce_memory=False, mask=None, true_array=None):
        """
        Verify that ExPSVM._compress_transform returns correct transformations from original space to the compressed
        linear model of the polynomial kernel.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=np.array([0]),
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)

        es._set_transform()
        if mask is None:
            trans = es._compress_transform(x=std_arr, reduce_memory=reduce_memory, mask=False, output_dict=True)
        else:
            es.interaction_mask = mask
            trans = es._compress_transform(x=std_arr, reduce_memory=reduce_memory, mask=True, output_dict=True)

        if true_array is None:
            true_array = std_transf_dict

        for dim in np.arange(1, std_d + 1):
            assert np.all(trans[dim] == true_array[dim])

    def test_compressed_transform_memory(self, std_p, std_d, std_r, std_gamma, std_transf_dict, std_arr):
        """
        Verify that ExPSVM._compress_transform also returns the correct linear model when run with reduced memory.
        """
        self.test_compressed_transform(std_p, std_d, std_r, std_gamma, std_transf_dict, std_arr, reduce_memory=True,
                                       mask=None)

    def test_compressed_transform_mask(self, std_p, std_d, std_r, std_gamma, std_transf_dict, std_arr, std_mask):
        """
        Verify that ExPSVM._compress_transform also returns correct linear models when run with mask.
        """
        true_array = {1: np.array([]), 2: np.array([[1, 3], [16, 24]])}
        self.test_compressed_transform(std_p, std_d, std_r, std_gamma, std_transf_dict, std_arr, reduce_memory=False,
                                       mask=std_mask, true_array=true_array)

    def test_poly_coef(self, std_p, std_d, std_r, std_gamma):
        """
        Verify that ExPSVM._poly_coef are correctly set after running ExPSVM._set_transform().
        """
        es = exp.ExPSVM(sv=None, dual_coef=None,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)
        es._set_transform()
        assert es._poly_coef == {1: 2, 2: 1}

    def test_transform_svm(self, std_p, std_d, std_r, std_arr,
                           std_dual_coef, std_lin_model, std_gamma,
                           mask=None):
        """
        Verify that a generic polynomial SVM model is correctly transformed into a linear model.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)

        if mask is None:
            es.transform_svm()
            assert np.all(es.linear_model == std_lin_model)
        else:
            es.interaction_mask = mask
            es.transform_svm(mask=True)
            assert np.all(es.linear_model == std_lin_model[mask])

    def test_transform_svm_mask(self, std_p, std_d, std_r, std_arr,
                                std_mask, std_dual_coef, std_lin_model,
                                std_gamma):
        """
        Verify that a generic polynomial SVM model is correctly transformed into a linear model when using a mask.
        """
        self.test_transform_svm(std_p, std_d, std_r, std_arr,
                                std_dual_coef, std_lin_model, std_gamma,
                                mask=std_mask)

    def test_polynomial_kernel(self, std_p, std_d, std_r, std_arr, std_gamma):
        """
        Verify that polynomial_kernel(), a help function in this test suite, produce correct Gram matrices.
        """
        assert np.all(
            polynomial_kernel(std_arr, std_arr, r=std_r, d=std_d, gamma=std_gamma) == np.array(
                [[225, 1089], [1089, 6084]]))
        assert np.all(
            polynomial_kernel(std_arr[0, :], std_arr, r=std_r, d=std_d, gamma=std_gamma) == np.array([[225, 1089]]))
        assert np.all(
            polynomial_kernel(std_arr, std_arr[0, :], r=std_r, d=std_d, gamma=std_gamma) == np.array([[225], [1089]]))
        assert np.all(
            polynomial_kernel(std_arr[0, :], std_arr[0, :], r=std_r, d=std_d, gamma=std_gamma) == np.array([225]))

    def test_decision_fun(self, std_p, std_d, std_r, std_arr, std_gamma, std_intercept):
        """
        Verify that the decision function produce the correct values.

        Author's note: It was at this test that I first was able to test that compressing the high-dimensional linear
        model produce identical result to the polynomial kernel. After thinking and writing equations as a hobby for two
        years, that this test passes feels wonderful.
        It works!
        """
        dual_coef = np.array([[1.]])  # Set to 1 to ignore effect of slack and class labels.
        sv = std_arr[1:2, :]
        es = exp.ExPSVM(sv=sv, dual_coef=dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=std_intercept)
        es.transform_svm()
        tol = 1e-12

        # When comparing the decision function between the polynomial kernel and
        # our decision function, we should add the SVM intercept and subtract
        # r^d. The latter term comes from the kernel expansion but is removed
        # due to the KKT conditions.
        constant = std_intercept - std_r ** std_d

        # Test single vector
        arr = rng.standard_normal(size=(1, 3))
        assert np.abs(es.decision_function(arr) -
                      (polynomial_kernel(sv, arr, r=std_r, d=std_d, gamma=std_gamma) + constant)
                      ) < tol

        # Test 2d array
        arr = rng.standard_normal(size=(2, 3))
        assert np.all(np.abs(es.decision_function(arr) -
                             (polynomial_kernel(sv, arr, r=std_r, d=std_d, gamma=std_gamma) + constant)
                             ) < tol)

        # Test 3d array. Should raise value error
        arr = rng.standard_normal(size=(2, 3, 1))
        with pytest.raises(ValueError):
            es.decision_function(arr)

    def test_decision_function_components(self, std_p, std_d, std_r, std_arr,
                                          std_dual_coef, std_idx, std_mask,
                                          std_intercept, std_lin_model,
                                          std_transf_dict, std_gamma):
        """
        Verify that the correct components of the decision function is returned. These components describe the
        importance of each feature when classifying a new observation.
        """
        # Test get components of the decision function
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=std_intercept)
        es.transform_svm()
        # const = std_r**std_d + std_intercept
        const = std_intercept

        true_components = np.multiply(exp.dict2array(std_transf_dict), np.transpose(std_lin_model))
        true_components = np.concatenate((np.array([[const], [const]]), true_components), axis=1)
        true_feat = np.concatenate((np.array(['intercept']), std_idx))
        test_comp, test_feat = es.decision_function_components(std_arr, output_interaction_names=True)
        assert np.all(test_comp == true_components)
        assert np.all(test_feat == true_feat)

        # Test components of decision function with mask
        es.interaction_mask = std_mask
        es.transform_svm(mask=True)
        true_components = np.multiply(exp.dict2array(std_transf_dict), np.transpose(std_lin_model))
        true_components = true_components[:, std_mask]
        true_components = np.concatenate((np.array([[const], [const]]), true_components), axis=1)
        true_feat = np.concatenate((np.array(['intercept']), std_idx[std_mask]))
        test_comp, test_feat = es.decision_function_components(std_arr, output_interaction_names=True, mask=True)
        assert np.all(test_comp == true_components)
        assert np.all(test_feat == true_feat)

        # Test same as previous but don't use mask=True in es.decision_function.
        # Function should realize that the linear model is already masked, so any
        # calculation of the decision model should automatically use mask as well.
        test_comp, test_feat = es.decision_function_components(std_arr, output_interaction_names=True)
        assert np.all(test_comp == true_components)
        assert np.all(test_feat == true_feat)

    def test_get_interactions(self, std_d, std_p, std_r, std_idx, std_gamma):
        """
        Verify that we are returned the correct interactions when using ExpSVM.get_interactions().
        """
        es = exp.ExPSVM(sv=None, dual_coef=None,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=None)
        es._set_transform()
        assert np.all(es.get_interactions(d=2) == std_idx[3:])

    def test_feature_importance(self, std_p, std_d, std_r, std_idx, std_gamma,
                                std_arr, std_dual_coef, std_mask, std_lin_model):
        """
        Verify that the importance, or magnitude in the linear model, of the interactions are correctly returned.
        """
        true_lin_model = np.squeeze(std_lin_model)
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=0)
        es.transform_svm()

        # Test unsorted with all features
        feat_imp, feat, _ = es.feature_importance(sort=False, include_intercept=False)
        assert np.all(feat_imp == true_lin_model)
        assert np.all(feat == std_idx)

        # Test unsorted with all features and intercept
        feat_imp, feat, _ = es.feature_importance(sort=False, include_intercept=True)
        assert np.all(feat_imp == np.append([0], true_lin_model))
        assert np.all(feat == np.append(['intercept'], std_idx))

        # Test sorted with all features
        feat_imp, feat, _ = es.feature_importance(include_intercept=False)
        sort_order = np.argsort(true_lin_model)[::-1]
        assert np.all(feat_imp == true_lin_model[sort_order])
        assert np.all(feat == std_idx[sort_order])

        # Test sorted with mask
        es.interaction_mask = std_mask
        feat_imp, feat, _ = es.feature_importance(mask=True, include_intercept=False)
        sort_order = np.argsort(true_lin_model[std_mask])[::-1]
        assert np.all(feat_imp == true_lin_model[std_mask][sort_order])
        assert np.all(feat == std_idx[std_mask][sort_order])

    def test_format_interaction_names(self, std_p, std_idx):
        """
        Check that formatting of interaction strings returns correctly formatted strings.
        """
        es = exp.ExPSVM(sv=None, dual_coef=None,
                        kernel_d=None, kernel_r=None, kernel_gamma=None,
                        p=std_p, intercept=None)
        formatted_strs = es.format_interaction_names(std_idx)
        assert np.all(formatted_strs == np.array(['$x_{0}$', '$x_{1}$', '$x_{2}$', '$x_{0}^{2}$', '$x_{0}$$x_{1}$',
                                                  '$x_{0}$$x_{2}$', '$x_{1}^{2}$', '$x_{1}$$x_{2}$', '$x_{2}^{2}$']))

    def test_feature_selection(self, std_p, std_d, std_r, std_arr, std_gamma,
                               std_dual_coef, std_lin_model):
        """
        Verify that the correct features are selected when using ExPSVM.feature_selection() with when used with
        n_interactions, frac_interactions and frac_importance independently.
        """
        true_lin_model = np.squeeze(std_lin_model)
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=0)
        es.transform_svm()

        sorted_lm = np.sort(true_lin_model)[::-1]

        # Test n_feat
        mask = es.feature_selection(n_interactions=3)
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[0:3]))

        # Test frac_feat
        mask = es.feature_selection(frac_interactions=0.5)
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[0:4]))

        # Test frac_feat_imp
        frac_feat_imp = 0.5
        mask = es.feature_selection(frac_importance=frac_feat_imp)
        cs = np.cumsum(sorted_lm) / (np.cumsum(sorted_lm)[-1])
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[cs < frac_feat_imp]))

    def test_set_mask(self, std_arr, std_dual_coef, std_d, std_gamma,
                      std_r, std_p, std_mask, std_lin_model):
        """
        Verify that the interaction mask is correctly set when using both feature selection and manual mask
        settings.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=0)
        es.transform_svm()

        # Test user-defined, boolean mask
        es.set_mask(mask=std_mask)
        assert np.all(es.interaction_mask == std_mask)

        # Test mask strings
        mask_strs = ['0,0', '0,2']
        es.set_mask(interaction_strs=mask_strs)
        assert np.all(es.interaction_mask == std_mask)

        # Test feature selection with n_interactions
        n_feat = 4
        true_bool = np.array([False, False, True,
                              False, False, True, False, True, True])
        es.set_mask(n_interactions=n_feat)
        assert np.all(es.interaction_mask == true_bool)

        # Test feature selection with fraction of interactions selected
        frac_feat = 0.5
        es.set_mask(frac_interactions=frac_feat)
        assert np.all(es.interaction_mask == true_bool)

        # Test feature selection with fraction of importance selected
        true_lin_model = np.squeeze(std_lin_model)
        sorted_lm = np.sort(true_lin_model)[::-1]
        frac_feat_imp = 0.5
        mask = es.feature_selection(frac_importance=frac_feat_imp)
        cs = np.cumsum(sorted_lm) / (np.cumsum(sorted_lm)[-1])
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[cs < frac_feat_imp]))

    def test_compare_sklearn_svc_artificial_data_2d(self):
        """
        Verify that the ExPSVM decision function and the Scikit-learn SVC decision function produce the same results
        on a toy dataset with 2 features.

        The dataset consists of one class being everything within the unit circle and the other class a ring concentric
        with the origin with minimum radius 1 and maximum radius 1.41.

        A tolerance of 1e-10 is used, i.e. the two decision functions should be within 1e-10 from each other on all
        test samples. The number of test samples is 50.
        """
        n_train_per_class = 125

        # Radii for rings
        r_min1 = 0.
        r_max1 = 1.
        r_min2 = 1.
        r_max2 = 1.41

        # Sample from classes
        phi_train1 = 2 * np.pi * rng.random(size=n_train_per_class)
        r_train1 = r_min1 + (r_max1 - r_min1) * rng.random(size=(n_train_per_class, 1))
        X_train1 = np.multiply(r_train1, np.transpose(np.array((np.cos(phi_train1), np.sin(phi_train1)))))

        phi_train2 = 2 * np.pi * rng.random(size=n_train_per_class)
        r_train2 = r_min2 + (r_max2 - r_min2) * rng.random(size=(n_train_per_class, 1))
        X_train2 = np.multiply(r_train2, np.transpose(np.array((np.cos(phi_train2), np.sin(phi_train2)))))

        X = np.concatenate((X_train1, X_train2), axis=0)
        y = np.concatenate((np.ones(n_train_per_class), -np.ones(n_train_per_class)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=10)
        # Fit SVM
        C = 0.9
        degree = 2
        gamma = 1. / np.pi
        r = np.sqrt(2)

        # Fit SVM
        kernel = 'poly'
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=r)
        model.fit(X_train, y_train)

        sv = model.support_vectors_
        dual_coef = np.squeeze(model.dual_coef_)
        intercept = model.intercept_[0]
        kernel_gamma = model._gamma

        es = exp.ExPSVM(sv=sv, dual_coef=dual_coef, intercept=intercept,
                        kernel_d=degree, kernel_r=r, kernel_gamma=kernel_gamma)
        es.transform_svm()

        sklearn_df = model.decision_function(X_test)
        expsvm_df = es.decision_function(X_test)
        tol = 1e-10
        assert np.all(np.abs(expsvm_df - sklearn_df) < tol)

    def test_compare_sklearn_svc_artificial_data_7d(self):
        """
        Verify that the ExPSVM decision function and the Scikit-learn SVC decision function produce the same results
        on a toy dataset with 7 features.

        The dataset is generated using Scikit-learn's make_classification, with 5 informative features,
        2 non-informative features, 2 classes and 3 clusters per class.

        A tolerance of 1e-10 is used, i.e. the two decision functions should be within 1e-10 from each other on all
        test samples. The number of test samples is 100.
        """
        n_samples = 1000
        n_features = 7
        n_informative = 5
        n_redundant = 2
        n_repeated = 0
        n_classes = 2
        n_clusters_per_class = 3
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=n_informative, n_redundant=n_redundant,
                                   n_repeated=n_repeated, n_classes=n_classes,
                                   n_clusters_per_class=n_clusters_per_class)
        # Convert labels to {-1,1}
        y[y == 0] = -1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)
        # Standardize data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        # Fit SVM
        C = 0.9
        degree = 3
        gamma = 'auto'
        r = np.sqrt(2)

        # Fit SVM
        kernel = 'poly'
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=r)
        model.fit(X_train, y_train)

        es = exp.ExPSVM(svc_model=model)

        es.transform_svm()
        sklearn_df = model.decision_function(X_test)
        expsvm_df = es.decision_function(X_test)
        tol = 1e-10
        assert np.all(np.abs(expsvm_df - sklearn_df) < tol)

    def test_compare_sklearn_svc_breast_cancer(self):
        """
        Verify that the ExPSVM decision function and the Scikit-learn SVC decision function produce the same results
        on the 30-dimensional breast cancer dataset.

        The Breast cancer dataset is downloaded using Scikit-learn and publicly available here:
        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        We are only interested in that the decision functions produce the same values, not that the classification
        performance is good.

        A tolerance of 1e-10 is used, i.e. the two decision functions should be within 1e-10 from each other on all
        test samples. The number of test samples is 100.
        """
        X, y = load_breast_cancer(return_X_y=True)
        y[y == 0] = -1
        # Split in training and test sets. Used to test decision function equality
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=100, random_state=10)

        # Standardize
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        # Fit SVM
        C = 1.0
        degree = 3
        gamma = 'scale'
        r = np.sqrt(2)
        model, es = create_sklearn_expsvm_models(X_train, y_train, C, degree, gamma, r)
        sklearn_df = model.decision_function(X_test)
        expsvm_df = es.decision_function(X_test)
        tol = 1e-10
        assert np.all(np.abs(expsvm_df - sklearn_df) < tol)

    def test_plot_model_bar(self, std_arr, std_dual_coef, std_d, std_gamma,
                      std_r, std_p):
        """
        Verify that plot_model_bar outputs a matplotlib figure.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=1)
        es.transform_svm()

        fig = es.plot_model_bar(n_features=2, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_sample_waterfall(self, std_arr, std_dual_coef, std_d, std_gamma,
                      std_r, std_p):
        """
        Verify that plot_sample_waterfall outputs a matplotlib figure.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=1)
        es.transform_svm()

        fig = es.plot_sample_waterfall(x=std_arr, n_features=2, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_sample_waterfall_degree(self, std_arr, std_dual_coef, std_d, std_gamma,
                      std_r, std_p):
        """
        Verify that plot_sample_waterfall_degree outputs a matplotlib figure.
        """
        es = exp.ExPSVM(sv=std_arr, dual_coef=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, kernel_gamma=std_gamma,
                        p=std_p, intercept=1)
        es.transform_svm()

        fig = es.plot_sample_waterfall_degree(x=std_arr, n_degree=2, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
