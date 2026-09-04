#pragma once

#include <cstddef>
#include <cstdint>

void smopa_matmul_s8(const int8_t* A, const int8_t* B,
                     size_t M, size_t N, size_t K, int32_t* C);