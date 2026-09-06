#include "matmul.hpp"
#include "ozaki-ii.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>
#include <sys/sysctl.h>

static int feat(const char* name)
{
    int v = 0;
    size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, nullptr, 0) == 0 ? v : 0;
}

static void dump_matrix(const char* path, const std::vector<double>& X)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(X.data()),
            (std::streamsize)(X.size() * sizeof(double)));
}

void sme_check()
{
    // --- evidence the SME unit is present and working -------------------
    printf("FEAT_SME=%d FEAT_SME2=%d SME_I8I32=%d\n",
           feat("hw.optional.arm.FEAT_SME"),
           feat("hw.optional.arm.FEAT_SME2"),
           feat("hw.optional.arm.SME_I8I32"));

    const size_t svl = sme_svl_bytes();
    printf("SVL = %zu bytes -> int32 tile %zux%zu, %zu int8 lanes\n",
           svl, svl / 4, svl / 4, svl);

    // micro-check: one-tile int8 gemm on ZA vs scalar arithmetic
    {
        const size_t n = svl / 4, k = 8;
        std::vector<int8_t> a(n * k), b(k * n);
        std::vector<int32_t> c(n * n);
        std::mt19937_64 rng(7);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (auto& v : a) v = (int8_t)d8(rng);
        for (auto& v : b) v = (int8_t)d8(rng);
        smopa_matmul_s8(a.data(), b.data(), n, n, k, c.data());
        size_t bad = 0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                int32_t acc = 0;
                for (size_t l = 0; l < k; l++)
                    acc += (int32_t)a[i * k + l] * (int32_t)b[l * n + j];
                bad += (acc != c[i * n + j]);
            }
        printf("SME SMOPA micro-check (%zux%zux%zu): %s\n\n",
               n, n, k, bad == 0 ? "PASS (bit-exact)" : "FAIL");
    }

}

int main()
{
    // Checking SME 
    sme_check();

    constexpr size_t M = 256;
    constexpr size_t N = 256;
    constexpr size_t K = 256;
    const size_t s = 14;

    printf("ozaki_scheme_ii: C(%zux%zu) = A(%zux%zu) * B(%zux%zu), s = %zu\n",
           M, N, M, K, K, N, s);
    
    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N), Cref(M * N, 0.0);

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for(auto& val : A) val = dist(rng);
    for(auto& val : B) val = dist(rng);

    // Call ozaki-ii !!
    ozaki_scheme_ii(A.data(), B.data(), M, N, K, s, C.data());

    // Ref run
    for(size_t i = 0; i < M; i++)
    {
        for(size_t k = 0; k < K; k++)
        {
            double aik = A[i * K + k];
            for(size_t j = 0; j < N; j++)
            {
                Cref[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    // --- error report ----------------------------------------------------
    double maxabs = 0, maxref = 0, maxrel = 0;
    for (size_t i = 0; i < M * N; i++) 
    {
        const double d = std::fabs(C[i] - Cref[i]);
        maxabs = std::fmax(maxabs, d);
        maxref = std::fmax(maxref, std::fabs(Cref[i]));
        maxrel = std::fmax(maxrel, d / std::fabs(Cref[i]));
    }
    printf("max |C - Cref|            = %.3e\n", maxabs);
    printf("normwise error (/max|C|)  = %.3e\n", maxabs / maxref);
    printf("max elementwise rel error = %.3e  (inflated by near-zero entries)\n",
           maxrel);

    // --- dump both matrices (row-major float64) ---------------------------
    dump_matrix("c_ref.bin", Cref);
    dump_matrix("c_ozaki.bin", C);
    printf("dumped c_ref.bin / c_ozaki.bin -- load with:\n"
           "  np.fromfile('c_ozaki.bin').reshape(%zu, %zu)\n", M, N);

    return 0;
}
