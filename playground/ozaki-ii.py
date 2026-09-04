import math
import numpy as np

def scalefpmat2int(X, beta, axis):
    max_abs = np.max(np.abs(X), axis=axis, keepdims=True)

    sig, e = np.frexp(max_abs)
    shift = beta - e

    X_scaled = np.ldexp(X, shift)
    X_int = np.rint(X_scaled).astype(np.int64)

    return X_int, shift.squeeze(axis)

def pick_moduli(s):
    m = 256
    moduli = []

    while len(moduli) < s and m >= 2:
        if all(math.gcd(m, prev) == 1 for prev in moduli):
            moduli.append(m)
        m -= 1
    assert len(moduli) == s
    return moduli

def center_mod(X, m):
    return ((X + m // 2) % m) - m // 2

def residual_matmul(A_int, B_int, m):
    Ar = center_mod(A_int, m).astype(np.int8)
    Br = center_mod(B_int, m).astype(np.int8)
    return Ar.astype(np.int32) @ Br.astype(np.int32)

def ozaki_scheme_ii(A, B, s=14):
    ma, na = A.shape
    mb, nb = B.shape

    moduli = pick_moduli(s)
    M = math.prod(moduli)

    k = int(math.log2(M // 2 - 1) - math.log2(na) // 2)
    kA = kB = min(k, 53)
    assert na * 2**(kA + kB) < M // 2

    A_int, shiftA = scalefpmat2int(A, kA, 1)
    B_int, shiftB = scalefpmat2int(B, kB, 0)

    weights = [(M // m) * pow((M // m), -1, m) for m in moduli] 

    Z = np.zeros([ma, nb], dtype=object)
    for t, m in enumerate(moduli):
        C_t = residual_matmul(A_int, B_int, m)
        C_t = C_t % m
        Z += C_t.astype(object) * weights[t]

    Z = Z % M
    C_int = center_mod(Z, M)
    C = np.ldexp(C_int.astype(np.float64), -(shiftA[:, None] + shiftB[None, :]))
    # C = np.zeros([ma, nb])
    # for i in range(ma):
    #     for j in range(nb):
    #         C[i, j] = np.ldexp(float(C_int[i, j]), -(shiftA[i] + shiftB[j]))

    return C

def main():
    seed = np.random.default_rng(0)

    A = seed.random(size=[4, 4])
    B = seed.random(size=[4, 4])

    C_ref = np.matmul(A, B)
    C_ozii = ozaki_scheme_ii(A, B)

    err = C_ref - C_ozii

    print(C_ref)
    print(C_ozii)
    print(err)

if __name__ == '__main__':
    main()