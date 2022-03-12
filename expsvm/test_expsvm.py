import numpy as np
import pytest
from expsvm import explain_svm as exp


class TestExPSVM:
    def test_create_index(self):
        p = 3
        d = 2
        tp = exp.TensorPerm(d, p)
        test_set = set(tp.create_unique_index())
        # true_set = {(0, 0), (0, 1), (0, 2),
        #             (1, 1), (1, 2), (2, 2)}
        true_set = {'0,0', '0,1', '0,2', '1,1', '1,2', '2,2'}
        sym_intersect = true_set.symmetric_difference(test_set)
        assert len(sym_intersect) == 0

    def test_count_occurrences(self):
        p = 3
        d = 2
        tp = exp.TensorPerm(d, p)
        tp.idx_unique = tp.create_unique_index()
        idx_count, unique_idx_count = tp._count_index_occurrences()
        compare = []
        for ind, idx in enumerate(tp.idx_unique):
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
        tp = exp.TensorPerm(d, p)
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
        tp = exp.TensorPerm(d, p)
        count_str = '1,2,2'
        n_perm = tp._count_perm(count_str)
        assert n_perm == 30

    def test_n_perm(self):
        p = 3
        d = 2
        tp = exp.TensorPerm(d, p)
        tp.n_perm()
        true_idx = ['0,0', '0,1', '0,2', '1,1', '1,2', '2,2']
        true_count = [1, 2, 2, 1, 2, 1]
        compare = []
        for ind, idx in enumerate(true_idx):
            tp_ind = tp.idx_unique.index(idx)
            compare.append(tp.idx_n_perm[tp_ind] == true_count[ind])
        assert all(compare)

    def test_multiplication_transform(self):
        p = 3
        d = 2
        r = 1
        es = exp.ExPSVM(sv=None, dual_coeff=0, kernel_d=d, kernel_r=r, p=p, intercept=None)
        es._multiplication_transform()
        true_idx = ['0', '1', '2', '0,0', '0,1', '0,2', '1,1', '1,2', '2,2']
        true_count = np.array([1, 1, 1, 1, 2, 2, 1, 2, 1])
        true_dim = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2])
        compare = []
        for ind, idx in enumerate(true_idx):
            es_ind = np.where(es.idx_unique == idx)[0]
            compare.append(np.all(es.perm_count[es_ind] == true_count[ind]))
            compare.append(np.all(es.idx_dim[es_ind] == true_dim[ind]))
        assert all(compare)

    def test_dict2array(self):
        di = {1: np.array([[1, 2, 3], [1, 2, 3]]),
              2: np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
        arr = exp.ExPSVM.dict2array(di)
        test_mat1 = np.all(arr == [[1, 2, 3, 1, 2, 3, 4, 6, 9], [1, 2, 3, 1, 2, 3, 4, 6, 9]])
        test_shape1 = arr.shape == (2,9)

        di = {1: np.array([[1, 2, 3]]),
              2: np.array([[1, 2, 3, 4, 6, 9]])}
        arr = exp.ExPSVM.dict2array(di)
        test_mat2 = np.all(arr == [[1, 2, 3, 1, 2, 3, 4, 6, 9]])
        test_shape2 = arr.shape == (1,9)
        assert all([test_mat1, test_shape1, test_mat2, test_shape2])

    def test_compressed_transform_full(self, reduce_memory=False):
        n_sv = 2
        p = 3
        d = 2
        r = 1
        arr = np.repeat(np.arange(1, p+1).reshape((-1, p)), n_sv, axis=0)
        es = exp.ExPSVM(sv=arr, dual_coeff=0, kernel_d=d, kernel_r=r, p=p, intercept=None)
        es._multiplication_transform()
        transf = es._compress_transform(x=arr, reduce_memory=False, mask=False)
        true_idx = {1: np.array([[0], [1], [2]]),
                    2: np.array([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])}
        true_array = {1:np.array([[1, 2, 3], [1, 2, 3]]),
                      2:np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
        compare = []
        for dim in np.arange(1, d + 1):
            compare.append(np.all(transf[dim] == true_array[dim]))
        assert all(compare)

    def test_compressed_transform_memory(self):
        self.test_compressed_transform_full(reduce_memory=True)

    def test_get_dim_idx(self, mask=None, true_idx=None):
        p = 3
        d = 2
        r = 1
        es = exp.ExPSVM(sv=None, dual_coeff=0, kernel_d=d, kernel_r=r, p=p, intercept=None)
        es._multiplication_transform()

        test_d = 2
        if true_idx is None:
            true_idx = ['0,0', '0,1', '0,2', '1,1', '1,2', '2,2']

        if mask is not None:
            es.mask_idx = mask
            test_idx = es.get_dim_idx(d=test_d, mask=True)
        else:
            test_idx = es.get_dim_idx(d=test_d, mask=False)
        assert np.all(test_idx == true_idx)

    def test_get_dim_idx_mask(self):
        mask = np.array([False, False, False, True, False, True, False, False, False])
        true_idx = ['0,0', '0,2']
        self.test_get_dim_idx(mask=mask, true_idx=true_idx)

    def test_poly_coef(self):
        p = 3
        d = 2
        r = 1
        es = exp.ExPSVM(sv=None, dual_coeff=None, kernel_d=d, kernel_r=r, p=p, intercept=None)
        true_poly = {1:2, 2:1}
        compare = []
        for dim in np.arange(1,d+1):
            compare.append(es.poly_coef[dim] == true_poly[dim])
        assert all(compare)

    def test_dekernelize_model(self):
        p = 3
        d = 2
        r = 1
        dual_coef = np.array([[10],[-0.1]])

        arr = np.array([[1,2,3],[4,5,6]])
        es = exp.ExPSVM(sv=arr, dual_coeff=dual_coef, kernel_d=d, kernel_r=r, p=p, intercept=None)

        es.dekernelize_model()

        true_lin_model = np.array([19.2, 39, 58.8,
                                   8.4, 36, 55.2, 37.5, 114, 86.4])
        assert np.all(es.linear_model == true_lin_model)

    def test_polynomial_kernel(self, arr=None, arr2=None, true_val=None):
        p = 3
        d = 2
        r = 1
        arr = np.array([[1., 2, 3], [4., 5, 6]])
        es = exp.ExPSVM(sv=None, dual_coeff=None, kernel_d=d, kernel_r=r, p=p, intercept=None)

        test_mat_mat = np.all(es.polynomial_kernel(arr, arr) == np.array([[225], [6084]]))
        test_vec_mat = np.all(es.polynomial_kernel(arr[0, :], arr) == np.array([[225], [1089]]))
        test_mat_vec = np.all(es.polynomial_kernel(arr, arr[0, :]) == np.array([[225], [1089]]))
        test_vec_vec = np.all(es.polynomial_kernel(arr[0, :], arr[0, :]) == np.array([[225]]))

        assert all([test_mat_mat, test_vec_mat, test_mat_vec, test_vec_vec])

    def test_linear_model_dot(self):
        """
        Verify that the compressed linear model produce the same result as the polynomial kernel
        :return:
        """
        p = 3
        d = 3
        r = 2
        intercept = 10
        dual_coef = np.array([[1.]])  # Set to 1 to ignore effect of slack and class labels.
        sv = np.array([[1., 2, 3]])
        es = exp.ExPSVM(sv=sv, dual_coeff=dual_coef, kernel_d=d, kernel_r=r, p=p, intercept=intercept)

        es.dekernelize_model()

        arr1 = np.array([[4.,5,6]])
        arr2 = np.array([[4.,5,6],[7.,8,9]])
        arr3 = np.random.randn(1,3)

        # Test with single observations
        test_dec_val1 = es._linear_model_dot(arr1)
        true_dec_val1 = es.polynomial_kernel(sv, arr1)
        # Test with two observations
        test_dec_val2 = es._linear_model_dot(arr2)
        true_dec_val2 = es.polynomial_kernel(sv, arr2)
        # Test that the two functions produce the same result within tolerance
        tol = 1e-12
        test_dec_val3 = es._linear_model_dot(arr3)
        true_dec_val3 = es.polynomial_kernel(sv, arr3)

        assert all([test_dec_val1 == true_dec_val1,
                    np.all(test_dec_val2 == true_dec_val2),
                    np.abs(test_dec_val3 - true_dec_val3) < tol])
