import numpy as np
from itertools import permutations
from functools import reduce

def get_permutations(n: int):
    mats = []
    elabels = []
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=np.int64)
            E[i, j] = 1
            elabels.append(f"E_{i+1}_{j+1}")
            mats.append(E)
    return np.array(mats), np.array(elabels)

def get_E_labels(indices: list, elabels):
    return elabels[indices]

def elementary_sum_with_metadata(n=3, verbose=True):
    # Build the n^2 elementary matrices in lexicographic order
    mats, elabels = get_permutations(n)

    assert len(mats)==n*n  # Should be True

    total = np.zeros((n, n), dtype=np.int64)
    count_nz = 0
    nz_permutations = []
    e_labels = []
    all_perm = list( permutations(range(n * n)) )
    for sigma in all_perm:
        prod = reduce(np.matmul, (mats[k] for k in sigma))

        if prod.any():
            count_nz += 1
            nz_permutations.append(sigma)
            #print(type(sigma), type(list(sigma)), list(sigma))
            e_labels.append(get_E_labels(list(sigma), elabels))

        total += prod

        if verbose:
            if len(all_perm) % 10000 == 0:
                print(f"Processed {count_nz} nonzero permutations...")
    return total, count_nz, nz_permutations, e_labels

if __name__ == "__main__":
    import sys
    dim = int(sys.argv[1])

    #if return_nz_counts or return_nz_perm:
    #    from elementary_sum_with_metadata import elementary_sum_with_metadata as func
    #elif pure_compute:
    #    from elementary_sum import elementary_sum as func

    S, nz, perm, elabels = elementary_sum_with_metadata(dim)        # 9! = 362,880 permutations
    print("Sum S =", S)
    print("Number of nonzero permutations:", nz)
    print("Permutations giving non-zero product:")
    print(*elabels, sep="\n")
