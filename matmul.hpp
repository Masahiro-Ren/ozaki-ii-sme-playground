#pragma once

#include <cstddef>
#include <cstdint>

size_t sme_svl_bytes();

void smopa_matmul_s8(const int8_t* A, const int8_t* B,
                     size_t M, size_t N, size_t K, int32_t* C);