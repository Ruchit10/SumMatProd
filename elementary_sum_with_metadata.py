import numpy as np
from itertools import permutations
from functools import reduce
import multiprocessing as mp

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

# Module-level globals used by worker processes (set via initializer)
_worker_mats = None
_worker_elabels = None

def _init_worker(mats, elabels):
    global _worker_mats, _worker_elabels
    _worker_mats = mats
    _worker_elabels = elabels

def _process_chunk(sigmas):
    """Process a list of permutations and return partial results."""
    n = _worker_mats.shape[1]
    partial_total = np.zeros((n, n), dtype=np.int64)
    count_nz = 0
    nz_permutations = []
    e_labels = []
    for sigma in sigmas:
        prod = reduce(np.matmul, (_worker_mats[k] for k in sigma))
        if prod.any():
            count_nz += 1
            nz_permutations.append(sigma)
            e_labels.append(get_E_labels(list(sigma), _worker_elabels))
        partial_total += prod
    return partial_total, count_nz, nz_permutations, e_labels

def elementary_sum_with_metadata(n=3, verbose=True, num_workers=None):
    # Build the n^2 elementary matrices in lexicographic order
    mats, elabels = get_permutations(n)

    assert len(mats) == n * n  # Should be True

    all_perm = list(permutations(range(n * n)))

    if num_workers is None:
        num_workers = mp.cpu_count()

    # Divide permutations into chunks — more chunks than workers for better load balancing
    chunk_size = max(1, len(all_perm) // (num_workers * 4))
    chunks = [all_perm[i:i + chunk_size] for i in range(0, len(all_perm), chunk_size)]

    if verbose:
        print(f"Processing {len(all_perm)} permutations across {num_workers} workers ({len(chunks)} chunks)...")

    with mp.Pool(num_workers, initializer=_init_worker, initargs=(mats, elabels)) as pool:
        results = pool.map(_process_chunk, chunks)

    # Aggregate partial results from all workers
    total = np.zeros((n, n), dtype=np.int64)
    count_nz = 0
    nz_permutations = []
    e_labels = []
    for partial_total, partial_count, partial_perm, partial_labels in results:
        total += partial_total
        count_nz += partial_count
        nz_permutations.extend(partial_perm)
        e_labels.extend(partial_labels)

    return total, count_nz, nz_permutations, e_labels

if __name__ == "__main__":
    import sys
    dim = int(sys.argv[1])

    S, nz, perm, elabels = elementary_sum_with_metadata(dim)        # 9! = 362,880 permutations
    print("Sum S =", S)
    print("Number of nonzero permutations:", nz)
    print("Permutations giving non-zero product:")
    print(*elabels, sep="\n")
