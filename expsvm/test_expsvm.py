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
        # true_idx = [(0, 0), (0, 1), (0, 2),
        #             (1, 1), (1, 2), (2, 2)]
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
        es = exp.ExPSVM(sv=None, dual_coeff=0, kernel_d=d, kernel_r=r, p=p)
        es._multiplication_transform()
        true_idx = {1:np.array([[0],[1],[2]]), 2:np.array([(0, 0), (0, 1), (0, 2),
                    (1, 1), (1, 2), (2, 2)])}
        true_idx = np.array([[0], [1], [2],
                             [0, 0], [0, 1], [0, 2],
                              [1, 1], [1, 2], [2, 2]])
        true_idx = ['0', '1', '2', '0,0', '0,1', '0,2', '1,1', '1,2', '2,2']
        true_count = {1:np.array([1,1,1]), 2:np.array([1, 2, 2, 1, 2, 1])}
        true_count = np.array([1, 1, 1, 1, 2, 2, 1, 2, 1])
        true_dim = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2])
        compare = []
        # for dim in np.arange(1,d+1):
        #     for ind, idx in enumerate(true_idx[dim]):
        #         es_ind = np.where(np.all(es.idx_unique[dim] == idx, axis=1))[0][0]
        #         compare.append(es.perm_count[dim][es_ind] == true_count[dim][ind])
        for ind, idx in enumerate(true_idx):
            es_ind = np.where(es.idx_unique == idx)[0]
            compare.append(np.all(es.perm_count[es_ind] == true_count[ind]))
            compare.append(np.all(es.idx_dim[es_ind] == true_dim[ind]))
        assert all(compare)


    def test_compressed_transform_full(self):
        n_sv = 2
        p = 3
        d = 2
        r = 1
        arr = np.repeat(np.arange(1, p+1).reshape((-1, p)), n_sv, axis=0)
        es = exp.ExPSVM(sv=arr, dual_coeff=0, kernel_d=d, kernel_r=r, p=p)
        es._multiplication_transform()
        print()
        # print(es.idx_unique)
        # print(es.idx_dim)
        # print(es.perm_count)
        transf = es._compress_transform(x=arr, memory_optimize=False, to_array=False)
        # true_idx = {1: np.array([[0], [1], [2]]),
        #             2: np.array([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])}
        true_idx = ['0', '1', '2', '0,0', '0,1', '0,2', '1,1', '1,2', '2,2']
        true_array = {1:np.array([[1, 2, 3], [1, 2, 3]]),
                      2:np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
        compare = []
        print()
        print(es.idx_unique)
        for dim in np.arange(1, d + 1):
            # print(transf[dim])
            for ind, idx in enumerate(true_idx):
                print(idx)
                es_ind = np.where(es.idx_unique == idx)[0][0]
                # es_ind = np.where(np.all(es.idx_unique[dim] == idx,axis=1))[0][0]
                print(ind, es_ind)
                compare.append(np.all(transf[dim][:,es_ind] == true_array[dim][:,ind]))
        assert all(compare)

    # def test_compressed_transform_memory(self):
    #     n_sv = 2
    #     p = 3
    #     d = 2
    #     r = 1
    #     arr = np.repeat(np.arange(1, p + 1).reshape((-1, p)), n_sv, axis=0)
    #     es = exp.ExPSVM(sv=arr, dual_coeff=0, kernel_d=d, kernel_r=r, p=p)
    #     es._multiplication_transform()
    #     transf = es._compress_transform(x=arr, memory_optimize=True, to_array=False)
    #     true_idx = {1: np.array([[0], [1], [2]]),
    #                 2: np.array([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])}
    #     true_array = {1: np.array([[1, 2, 3], [1, 2, 3]]),
    #                   2: np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
    #     compare = []
    #     for dim in np.arange(1, d + 1):
    #
    #         for ind, idx in enumerate(true_idx[dim]):
    #             es_ind = np.where(np.all(es.idx_unique[dim] == idx, axis=1))[0][0]
    #             compare.append(np.all(transf[dim][:, es_ind] == true_array[dim][:, ind]))
    #     assert all(compare)
    #
    # def test_dict2array(self):
    #     di = {1: np.array([[1, 2, 3], [1, 2, 3]]),
    #           2: np.array([[1, 2, 3, 4, 6, 9], [1, 2, 3, 4, 6, 9]])}
    #     arr = exp.ExPSVM.dict2array(di)
    #     assert np.all(arr == [[1, 2, 3, 1, 2, 3, 4, 6, 9], [1, 2, 3, 1, 2, 3, 4, 6, 9]])
    #
    # def test_dekernelize(self):
    #     n_sv = 2
    #     p = 3
    #     d = 2
    #     r = 1
    #     dual_coef = np.array([[10, -0.1]])
    #
    #     arr = np.repeat(np.arange(1, p + 1).reshape((-1, p)), n_sv, axis=0)
    #     es = exp.ExPSVM(sv=arr, dual_coeff=dual_coef, kernel_d=d, kernel_r=r, p=p)
    #
    #     es.dekernelize()
    #




