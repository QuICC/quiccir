#pragma once

#include <cstdint>
#include <string>

template<typename T, std::size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

using mem3_t = MemRefDescriptor<double, 3>;

template<typename Tdata, typename Tmeta, std::size_t N>
struct ViewDescriptor {
  Tmeta dims[N];
  Tmeta *pos;
  Tmeta posSize;
  Tmeta *coo;
  Tmeta cooSize;
  Tdata *data;
  Tmeta dataSize;
};

using view3_t = ViewDescriptor<double, std::uint32_t, 3>;