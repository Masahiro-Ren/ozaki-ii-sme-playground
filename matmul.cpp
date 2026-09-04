#include "matmul.hpp"

#include <arm_sme.h>

__arm_locally_streaming __arm_new("za")
void smopa_matmul_s8(const int8_t* A, const int8_t* B,
                     size_t M, size_t N, size_t K, int32_t* C)
{
    const size_t T = svcntw();          // tile dim: SVL/32 (16 on M5)
    const svbool_t pg8 = svptrue_b8();

    // SVL is at most 2048 bits = 256 bytes, so fixed buffers suffice
    int8_t apack[256], bpack[256];

    for(size_t i0 = 0; i0 < M; i0 += T)
    {
        const size_t mrows = (M - i0 < T) ? M - i0 : T;
        for(size_t j0 = 0; j0 < N; j0 += T)
        {
            const size_t ncols = (N - j0 < T) ? N - j0 : T;
            const svbool_t pcol = svwhilelt_b32((uint32_t)0, (uint32_t)ncols);

            svzero_za();
            for(size_t c = 0; c < K; c += 4)
            {
                const size_t kc = (K - c < 4) ? K - c : 4;
                // apack[4i+t] = A[i0+i][c+t], bpack[4j+t] = B[c+t][j0+j]
                for(size_t i = 0; i < T; i++)
                    for(size_t t = 0; t < 4; t++)
                        apack[4 * i + t] = (i < mrows && t < kc)
                            ? A[(i0 + i) * K + c + t] : 0;
                for(size_t j = 0; j < T; j++)
                    for(size_t t = 0; t < 4; t++)
                        bpack[4 * j + t] = (j < ncols && t < kc)
                            ? B[(c + t) * N + j0 + j] : 0;

                svint8_t a = svld1_s8(pg8, apack);
                svint8_t b = svld1_s8(pg8, bpack);
                svmopa_za32_s8_m(0, pg8, pg8, a, b);
            }

            for(size_t i = 0; i < mrows; i++)
                svst1_hor_za32(0, (uint32_t)i, pcol,
                               C + (i0 + i) * N + j0);
        }
    }
}
