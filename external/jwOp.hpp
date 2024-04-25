#pragma once

#include <cstdint>
#include <string>

#include "Quiccir-c/Utils.h"

// Mock operation backend (dense batched matmul)
template <class T>
void cpu_op(T *c, const T *a, const T *b, const std::size_t L, const std::size_t M, const std::size_t K, const std::size_t N);

/// @brief mockup class for quadrature op backend
class JWOp
{
private:
  view3_t _opData;
public:
  std::string _str;
  view3_t& getOp();
  void apply(view3_t* pUval, view3_t* pUmod);
};

// Mock operation backend (dense batched matmul)
template <class T>
void cpu_op(T *c, const T *a, const T *b, const std::size_t L, const std::size_t M, const std::size_t K, const std::size_t N)
{
  for (std::size_t l = 0; l < L; ++l)
  {
    for (std::size_t m = 0; m < M; ++m)
    {
      for (std::size_t n = 0; n < N; ++n)
      {
        auto lmn = l*M*N + m*N + n;
        T acc = 0;
        for (std::size_t k = 0; k < K; ++k)
        {
          // row major
          auto lmk = l*M*K + m*K + k;
          auto lkn = l*K*N + k*N + n;
          acc += a[lmk]*b[lkn];
        }
        c[lmn] = acc;
      }
    }
  }
}
