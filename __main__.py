from expsvm import explain_svm as exp
import numpy as np
p = 4
d = 3
# idx = exp.create_unique_index(p, d)
# print(idx)
# labels, counts_unique = exp.count_index_occurrences(idx)
# print(counts_unique)
# print(labels)
# perms = [exp.count_symmetry(count) for count in counts_unique]
# print(perms)
# sym_count = exp.map_symmetry(labels, counts_unique)
# print(sym_count)

n = np.arange(4)
idx = [(0,1), (0,2)]
print(np.array(idx))
