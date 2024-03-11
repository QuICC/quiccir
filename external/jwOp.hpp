#pragma once

#include <cstdint>
#include <string>

#include "utils.hpp"

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