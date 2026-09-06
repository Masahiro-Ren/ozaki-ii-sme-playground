#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <iostream>

std::vector<int> pick_moduli(size_t s);

void center_mod(const int64_t* X, size_t rows, size_t cols, int m,
                int8_t* out);

std::vector<int> scalefpmat2int(const double* X, size_t M, size_t N,
                                int beta, int axis, int64_t* out);

void residual_matmul(const int64_t* A_int, const int64_t* B_int,
                     size_t M, size_t N, size_t K, int m, int32_t* C);

void ozaki_scheme_ii(const double* A, const double* B, size_t M, size_t N,
                     size_t K, size_t s, double* C);
