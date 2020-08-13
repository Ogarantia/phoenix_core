/**
 * @file backend.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Backend interface declaration
 * This file
 * (1) defines basic datatypes required for the whole stack and
 * (2) declares operations to be implemented in every backend.
 * To be explicitly included only in the backend level.
 * @copyright Copyright (c) 2020 UpStride.io
 */

#pragma once

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace upstride {

/**
 * @brief Backend enumeration
 */
namespace device {

typedef struct {
} CPU;      //!< CPU oneDNN backend
typedef struct {
} CUDA;     //!< CUDA/cuDNN GPU backend

}  // namespace device

/**
 * @brief A fairly generic integer tuple
 */
typedef std::vector<int32_t> IntTuple;

/**
 * @brief A lightweight pair of integer numbers
 */
class IntPair {
   public:
    int x, y;

    IntPair() : x(0), y(0) {}
    IntPair(int val) : x(val), y(val) {}
    IntPair(int x, int y) : x(x), y(y) {}

    inline IntPair operator+(const IntPair& another) const {
        return IntPair(x + another.x, y + another.y);
    }

    inline bool operator==(const IntPair& another) const {
        return x == another.x && y == another.y;
    }
};

/**
 * @brief Base class of a context shared between different operations
 */
class Context {
    const int typeDimensions;

   protected:
    Context(const int td) : typeDimensions(td){};
};

/**
 * @brief Scalar 2D convolution operation
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarConv2DFunctor;

/**
 * @brief Scalar 2D convolution operation gradient
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarConv2DGradFunctor;

}  // namespace upstride