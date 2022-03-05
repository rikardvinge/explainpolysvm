from expsvm import explain_svm as exp
p = 4
d = 3
idx = exp.create_unique_index(p, d)
print(idx)
counts_unique, labels = exp.count_index_occurrences(idx)
print(counts_unique)
print(labels)
perms = [exp.count_symmetry(count) for count in counts_unique]
print(perms)
sym_count = exp.map_symmetry(labels, counts_unique)
print(sym_count)
