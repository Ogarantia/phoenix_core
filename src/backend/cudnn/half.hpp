#pragma once
#include <cuda_fp16.h>

namespace upstride {
namespace cudnn {

typedef __half half;

template <typename T>
constexpr bool isHalfFloat() {
    return false;
}

template <>
constexpr bool isHalfFloat<half>() {
    return true;
}

}  // namespace cudnn
}  // namespace upstride