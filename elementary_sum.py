def elementary_sum(n=3):
    # Build the n^2 elementary matrices in lexicographic order
    mats = []
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=np.int64)
            E[i, j] = 1
            mats.append(E)

    total = np.zeros((n, n), dtype=np.int64)
    for sigma in permutations(range(n * n)):
        prod = reduce(np.matmul, (mats[k] for k in sigma))
        total += prod
    return total

if __name__ == "__main__":
    # parallelise
