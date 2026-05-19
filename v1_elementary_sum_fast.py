import numpy as np
from math import factorial

def N(n):
    # n^(n-2) * n! * ((n-1)!)^(n-1)
    return (n ** (n - 2)) * factorial(n) * (factorial(n - 1) ** (n - 1))

def S(n):
    return N(n) * np.eye(n, dtype=object)

if __name__ == "__main__":
    # call signature: python v1_elementary_sum_fast.py <dimension>
    import sys
    dim = int(sys.argv[1])

    print(S(dim))      # 72 * I_3
    print(N(dim))      # 72  (closed form for n = 3)