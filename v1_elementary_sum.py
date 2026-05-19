import numpy as np
from itertools import permutations
from functools import reduce

def elementary_sum(n=3):
    # Build the n^2 elementary matrices in lexicographic order
    mats = []
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=np.int64)
            E[i, j] = 1
            mats.append(E)

    assert len(mats)==n*n  # Should be True

    total = np.zeros((n, n), dtype=np.int64)
    count_nonzero = 0
    for sigma in permutations(range(n * n)):
        prod = reduce(np.matmul, (mats[k] for k in sigma))
        if prod.any():
            count_nonzero += 1
        total += prod
        if count_nonzero % 10000 == 0:
            print(f"Processed {count_nonzero} nonzero permutations...")
    return total, count_nonzero

import sys
dim = int(sys.argv[1])

if __name__ == "__main__":
    S, nz = elementary_sum(dim)        # 9! = 362,880 permutations
    print("Sum S =")
    print(S)
    print("Nonzero permutations:", nz)