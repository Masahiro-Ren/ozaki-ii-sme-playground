#include "ozaki-ii.hpp"
#include "matmul.hpp"

using std::vector;

vector<int> pick_moduli(size_t s)
{
    vector<int> moduli;
    for(int m = 256; moduli.size() < s && m >= 2; --m)
    {
        bool coprime = true;
        for(int prev : moduli)
            if(std::gcd(m, prev) != 1) {coprime = false; break;}

        if(coprime)
            moduli.push_back(m);
    }

    assert(moduli.size() == s);
    return moduli;
}

static int mod_inverse(int x, int m)
{
    int r0 = m;
    int r1 = x;
    int t0 = 0;
    int t1 = 1;

    while(r1 != 0)
    {
        int q = r0 / r1;
        int r = r0 - q * r1; r0 = r1; r1 = r;
        int t = t0 - q * t1; t0 = t1; t1 = t;
    }
    return t0 < 0 ? t0 + m : t0;
}

void center_mod(const int64_t* X, size_t rows, size_t cols, int m, int8_t* out)
{
    const int half_m = m / 2;
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            size_t idx = i * cols + j;
            int r = (X[idx] + half_m) % m;
            if(r < 0) r += m;               // C++ % truncates: make it floor mod
            out[idx] = r - half_m;
        }
    }
}

vector<int> scalefpmat2int(const double* X, size_t M, size_t N, 
                    int beta, int axis,
                    int64_t* out)
{
    const size_t ngroups = (axis == 1) ? M: N;
    vector<double> maxs(ngroups, 0.0);

    for(size_t i = 0; i < M; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            size_t idx = i * N + j;
            size_t g = (axis == 1) ? i : j;
            maxs[g] = std::max(maxs[g], std::abs(X[idx]));
        }
    }

    vector<int> shift(ngroups);
    for(size_t g = 0; g < ngroups; g++)
    {
        int e;
        std::frexp(maxs[g], &e);
        shift[g] = beta - e;
    }

    for(size_t i = 0; i < M; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            size_t idx = i * N + j;
            size_t g = (axis == 1) ? i : j;
            out[idx] = std::llrint(std::ldexp(X[idx], shift[g]));
        }
    }

    return shift;
}

void residual_matmul(const int64_t* A_int, const int64_t* B_int, 
                     size_t M, size_t N, size_t K,
                     int m, int32_t* C)
{
    assert(K < (size_t)1 << 17);

    vector<int8_t> Ar(M * K);
    vector<int8_t> Br(K * N);

    center_mod(A_int, M, K, m, Ar.data());
    center_mod(B_int, K, N, m, Br.data());

    smopa_matmul_s8(Ar.data(), Br.data(), M, N, K, C);
}

void ozaki_scheme_ii(const double* A, const double* B, size_t M, size_t N, size_t K,
                     size_t s, double* C)
{
    assert(s <= 14);
    vector<int> moduli = pick_moduli(s);

    __int128 MM = 1;
    for(int m : moduli)
        MM *= m;
    const __int128 half_MM = MM / 2;

    int k = (int)((std::log2((double)MM) - 1 - std::log2((double)K)) / 2);
    int kA, kB;
    kA = kB = std::min(k, 53);
    assert(((__int128)K << (kA + kB)) < half_MM);   // CRT uniqueness

    vector<int64_t> A_int(M * K);
    vector<int64_t> B_int(K * N);
    vector<int> shiftA = scalefpmat2int(A, M, K, kA, 1, A_int.data());
    vector<int> shiftB = scalefpmat2int(B, K, N, kB, 0, B_int.data());

    vector<__int128> W;
    for(int m : moduli)
    {
        __int128 Mt = MM / m;
        W.push_back(Mt * mod_inverse((Mt % m), m));
    }

    vector<__int128> Z(M * N, 0);
    vector<int32_t>  Ct(M * N);

    for(size_t t = 0; t < s; t++)
    {
        residual_matmul(A_int.data(), B_int.data(), M, N, K, moduli[t], Ct.data());
        for(size_t i = 0; i < M * N; i++)
        {
            int32_t u = Ct[i] % moduli[t];
            if(u < 0) u += moduli[t];
            Z[i] += (__int128)u * W[t];
        }
    }

    for(auto& z : Z)
    {
        z %= MM;
        if(z > half_MM) z -= MM;
    }

    for(size_t i = 0; i < M; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            size_t idx = i * N + j;
            C[idx] = std::ldexp((double)Z[idx], -(shiftA[i] + shiftB[j]));
        }
    }

}