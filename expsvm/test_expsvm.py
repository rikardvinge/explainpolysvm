import numpy as np
import pytest
from expsvm import explain_svm as exp


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
def std_mask():
    return np.array([False, False, False,
                     True, False, True, False, False, False])


class TestExPSVM:
    def test_create_index(self, std_d, std_p, std_idx):
        tp = exp.TensorUtil(std_d, std_p)
        test_set = set(tp.create_unique_index())
        true_set = set(std_idx[3:])
        sym_intersect = true_set.symmetric_difference(test_set)
        assert len(sym_intersect) == 0

    def test_count_occurrences(self, std_d, std_p):
        tp = exp.TensorUtil(std_d, std_p)
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
        tp = exp.TensorUtil(d, p)
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
        tp = exp.TensorUtil(d, p)
        count_str = '1,2,2'
        n_perm = tp._count_perm(count_str)
        assert n_perm == 30

    def test_n_perm(self, std_d, std_p, std_idx, std_perm_count):
        tu = exp.TensorUtil(std_d, std_p)
        idx_list, n_perm = tu.n_perm()

        # Check that all indexes exist
        assert set(idx_list) == set(std_idx[3:])

        tu_order = np.argsort(idx_list)
        idx_order = np.argsort(std_idx[3:])
        assert np.all(np.array(n_perm)[tu_order] == std_perm_count[3:][idx_order])

    def test_multiplication_transform(self, std_d, std_p, std_r,
                                      std_idx, std_perm_count, std_dim):
        es = exp.ExPSVM(sv=None, dual_coeff=0,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es._multiplication_transform()

        # Check that all indexes exist
        assert set(es.idx_unique) == set(std_idx)

        # Check that permutation counts and dimensions are correct
        # But first, order index
        es_order = np.argsort(es.idx_unique)
        idx_order = np.argsort(std_idx)
        assert np.all(es.perm_count[es_order] == std_perm_count[idx_order])
        assert np.all(es.idx_dim[es_order] == std_dim[idx_order])

    def test_dict2array(self, std_transf_dict):
        # Test with two samples
        arr1 = exp.ExPSVM.dict2array(std_transf_dict)

        # Test with single observation with format (1,p)
        di2 = {1: std_transf_dict[1][0:1, :],
               2: std_transf_dict[2][0:1, :]}
        arr2 = exp.ExPSVM.dict2array(di2)

        # Test with single observation with format (p,).
        di3 = {1: std_transf_dict[1][0, :],
               2: std_transf_dict[2][0, :]}
        arr3 = exp.ExPSVM.dict2array(di3)
        assert np.all(arr1 == [[1., 2., 3., 1., 2., 3., 4., 6., 9.],
                               [4, 5, 6, 16., 20., 24., 25., 30., 36.]])
        assert arr1.shape == (2, 9)
        assert np.all(arr2 == [[1., 2., 3., 1., 2., 3., 4., 6., 9.]])
        assert arr2.shape == (1, 9)
        assert np.all(arr3 == [1., 2., 3., 1., 2., 3., 4., 6., 9.])
        assert arr3.shape == (9,)

    def test_get_dim_mask_index(self, std_p, std_d, std_r, std_idx, std_mask, std_dim,
                                mask=None, true_idx=None):
        es = exp.ExPSVM(sv=None, dual_coeff=0,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es._multiplication_transform()
        es.mask_idx = std_mask

        # Test get all indices
        ind_full = es.get_dim_mask_index()
        assert np.all(ind_full == np.full(es.idx_unique.shape, True))

        # Test get dimension indices only
        ind_d = es.get_dim_mask_index(d=std_d)
        assert np.all(ind_d == (std_dim == 2))

        # Test get mask indices only
        ind_m = es.get_dim_mask_index(mask=True)
        assert np.all(ind_m == std_mask)

        # Test get dimension and mask indices
        es.mask_idx = np.array([True, False, False, True, False, True, False, False, False])
        ind_d_m = es.get_dim_mask_index(d=2, mask=True)
        assert np.all(ind_d_m == std_mask)

    def test_compressed_transform(self, std_p, std_d, std_r, std_transf_dict, std_arr,
                                  reduce_memory=False, mask=None, true_array=None):
        es = exp.ExPSVM(sv=std_arr, dual_coeff=0,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)

        es._multiplication_transform()
        if mask is None:
            trans = es._compress_transform(x=std_arr, reduce_memory=reduce_memory, mask=False)
        else:
            es.mask_idx = mask
            trans = es._compress_transform(x=std_arr, reduce_memory=reduce_memory, mask=True)

        if true_array is None:
            true_array = std_transf_dict

        for dim in np.arange(1, std_d + 1):
            assert np.all(trans[dim] == true_array[dim])

    def test_compressed_transform_memory(self, std_p, std_d, std_r, std_transf_dict, std_arr):
        self.test_compressed_transform(std_p, std_d, std_r, std_transf_dict, std_arr, reduce_memory=True, mask=None)

    def test_compressed_transform_mask(self, std_p, std_d, std_r, std_transf_dict, std_arr, std_mask):
        true_array = {1: np.array([]), 2: np.array([[1, 3], [16, 24]])}
        self.test_compressed_transform(std_p, std_d, std_r, std_transf_dict, std_arr, reduce_memory=False,
                                       mask=std_mask, true_array=true_array)

    def test_poly_coef(self, std_p, std_d, std_r):
        es = exp.ExPSVM(sv=None, dual_coeff=None,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        assert es.poly_coef == {1: 2, 2: 1}

    def test_transform_svm(self, std_p, std_d, std_r, std_arr, std_dual_coef, mask=None):
        true_lin_model = np.transpose(np.array([[19.2, 39, 58.8,
                                                 8.4, 36, 55.2, 37.5, 114, 86.4]]))
        es = exp.ExPSVM(sv=std_arr, dual_coeff=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)

        if mask is None:
            es.transform_svm()
            assert np.all(es.linear_model == true_lin_model)
        else:
            es.mask_idx = mask
            es.transform_svm(mask=True)
            assert np.all(es.linear_model == true_lin_model[mask])

    def test_transform_svm_mask(self, std_p, std_d, std_r, std_arr, std_mask, std_dual_coef):
        self.test_transform_svm(std_p, std_d, std_r, std_arr, std_dual_coef, mask=std_mask)

    def test_polynomial_kernel(self, std_p, std_d, std_r, std_arr):
        es = exp.ExPSVM(sv=None, dual_coeff=None,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)

        assert np.all(es.polynomial_kernel(std_arr, std_arr) == np.array([[225], [6084]]))
        assert np.all(es.polynomial_kernel(std_arr[0, :], std_arr) == np.array([[225], [1089]]))
        assert np.all(es.polynomial_kernel(std_arr, std_arr[0, :]) == np.array([[225], [1089]]))
        assert np.all(es.polynomial_kernel(std_arr[0, :], std_arr[0, :]) == np.array([[225]]))

    def test_linear_model_dot(self, std_p, std_d, std_r, std_arr, std_mask):
        """
        Verify that the compressed linear model produce the same result as the polynomial kernel

        Author's note: It was at this test, that I, after more than 2 years of thinking, finally got to test that rewriting the polynomial kernel SVM as a compressed linear model which can be directly related to feature importance. Felt great when the test passed! So, Yay!
        :return:
        """
        dual_coef = np.array([[1.]])  # Set to 1 to ignore effect of slack and class labels.
        sv = std_arr[1:2, :]
        es = exp.ExPSVM(sv=sv, dual_coeff=dual_coef,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es.transform_svm()
        es.mask_idx = std_mask

        arr1 = np.array([[4., 5, 6]])
        arr2 = np.array([[4., 5, 6], [7., 8, 9]])
        arr3 = np.random.randn(1, 3)

        # Test with single observations
        assert np.all(es._linear_model_dot(arr1, reduce_memory=False, mask=False) == es.polynomial_kernel(sv, arr1))

        # Test with two observations
        assert np.all(es._linear_model_dot(arr2, reduce_memory=False, mask=False) == es.polynomial_kernel(sv, arr2))

        # Test that the two functions produce the same result within tolerance
        tol = 1e-12
        assert np.all(np.abs(
            es._linear_model_dot(arr3, reduce_memory=False, mask=False) - es.polynomial_kernel(sv, arr3)) < tol)

        # Test with reduced memory
        assert np.all(es._linear_model_dot(arr2, reduce_memory=True, mask=False) == es.polynomial_kernel(sv, arr2))

        # Test with mask
        true_val = np.array([[(1 * sv[0, 0] ** 2) * (arr1[0, 0] ** 2) +
                              (2 * sv[0, 0] * sv[0, 2]) * (arr1[0, 0] * arr1[0, 2]) +
                              std_r ** std_d]])
        assert np.all(es._linear_model_dot(arr1, reduce_memory=False, mask=True) == true_val)

    def test_decision_fun(self, std_p, std_d, std_r, std_arr, std_mask, std_intercept):
        """
        Verify that the decision function produce the correct values.
        """
        dual_coef = np.array([[1.]])  # Set to 1 to ignore effect of slack and class labels.
        sv = std_arr[1:2, :]
        es = exp.ExPSVM(sv=sv, dual_coeff=dual_coef,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=std_intercept)
        es.transform_svm()
        arr = np.random.randn(1, 3)
        tol = 1e-12
        assert np.all(np.abs(
            es._linear_model_dot(arr) - es.polynomial_kernel(sv, arr)) < tol)

    def test_get_idx_unique(self, std_d, std_p, std_r, std_idx):
        es = exp.ExPSVM(sv=None, dual_coeff=None,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es._multiplication_transform()
        assert np.all(es.get_idx_unique(d=2) == std_idx[3:])

    def test_feature_importance(self, std_p, std_d, std_r, std_idx,
                                std_arr, std_dual_coef, std_mask):
        true_lin_model = np.array([19.2, 39, 58.8,
                                   8.4, 36, 55.2, 37.5, 114, 86.4])
        es = exp.ExPSVM(sv=std_arr, dual_coeff=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es.transform_svm()

        # Test unsorted with all features
        feat_imp, feat, _ = es.feature_importance(sort=False)
        assert np.all(feat_imp == true_lin_model)
        assert np.all(feat == std_idx)

        # Test sorted with all features
        feat_imp, feat, _ = es.feature_importance()
        sort_order = np.argsort(true_lin_model)[::-1]
        assert np.all(feat_imp == true_lin_model[sort_order])
        assert np.all(feat == std_idx[sort_order])

        # Test sorted with mask
        es.mask_idx = std_mask
        feat_imp, feat, _ = es.feature_importance(mask=True)
        sort_order = np.argsort(true_lin_model[std_mask])[::-1]
        assert np.all(feat_imp == true_lin_model[std_mask][sort_order])
        assert np.all(feat == std_idx[std_mask][sort_order])

    def test_feature_selection(self, std_p, std_d, std_r, std_arr, std_dual_coef):
        true_lin_model = np.array([19.2, 39, 58.8,
                                   8.4, 36, 55.2, 37.5, 114, 86.4])
        es = exp.ExPSVM(sv=std_arr, dual_coeff=std_dual_coef,
                        kernel_d=std_d, kernel_r=std_r, p=std_p,
                        intercept=None)
        es.transform_svm()

        sorted_lm = np.sort(true_lin_model)[::-1]

        # Test n_feat
        mask = es.feature_selection(n_feat=3)
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[0:3]))

        # Test frac_feat
        mask = es.feature_selection(frac_feat=0.5)
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[0:4]))

        # Test frac_feat_imp
        frac_feat_imp = 0.5
        mask = es.feature_selection(frac_feat_imp=frac_feat_imp)
        cs = np.cumsum(sorted_lm)/(np.cumsum(sorted_lm)[-1])
        assert np.all(set(es.linear_model[mask, 0]) == set(sorted_lm[cs < frac_feat_imp]))

