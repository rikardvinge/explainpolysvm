import pytest
from expsvm import explain_svm as exp


class TestExPSVM:
    def test_create_index(self):
        p = 3
        d = 2
        tp = exp.TensorPerm(d, p)
        test_set = tp.create_unique_index()
        true_set = {(0, 0), (0, 1), (0, 2),
                    (1, 1), (1, 2), (2, 2)}
        sym_intersect = true_set.symmetric_difference(test_set)
        assert len(sym_intersect) == 0

    def test_count_occurrences(self):
        p = 3
        d = 2
        tp = exp.TensorPerm(d, p)
        tp.idx_unique = tp.create_unique_index()
        idx_count, unique_idx_count = tp._count_index_occurrences()
        test_lst = []
        for ind, idx in enumerate(tp.idx_unique):
            if idx[0] == idx[1]:
                test_lst.append(idx_count[ind] == '2')
            else:
                test_lst.append(idx_count[ind] == '1,1')
        assert all(test_lst)

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
        (i,j,j,k,k), (k,i,j,j,k), (k,k,i,j,j), ...
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
        true_idx = [(0, 0), (0, 1), (0, 2),
                    (1, 1), (1, 2), (2, 2)]
        true_count = [1, 2, 2, 1, 2, 1]
        test_lst = []
        for ind, idx in enumerate(true_idx):
            tp_ind = tp.idx_unique.index(idx)
            test_lst.append(tp.idx_n_perm[tp_ind] == true_count[ind])
        assert all(test_lst)
