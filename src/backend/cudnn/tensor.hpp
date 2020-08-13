/**
 * @file tensor.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Tensor arithmetics and allocation implementation using CUDA
 *
 * @copyright Copyright (c) 2020 UpStride
 */
#pragma once

#include "../backend.hpp"
#include "kernels.hpp"

namespace upstride {
template <>
struct TensorManipulations<device::CUDA> {
    template <typename T>
    void accumulateAdd(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, const Shape& shape) {
        // TODO
    }

    template <typename T>
    void accumulateSub(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, const Shape& shape) {
        // TODO
    }
};

/**
 * @brief Implementation of allocated Tensor for CUDA
 * This tensor manages its own memory. Its instantiation/destruction implies memory allocation/disposition. Do not abuse with this, use with care!
 * @tparam T scalar datatype of tensor entries
 */
template <typename T>
class AllocatedTensor<device::CUDA, T> : public Tensor<device::CUDA, T> {
    AllocatedTensor(const Shape&, T*) = delete;  // deleting Tensor constructor allowing to wrap an external pointer

   public:
    AllocatedTensor(const Shape& shape) : Tensor<device::CUDA, T>(shape, nullptr) {
        // TODO: allocate the memory and assign it to the tensor pointer
    }

    ~AllocatedTensor() {
        // TODO: free memory
    }
};
}  // namespace upstride