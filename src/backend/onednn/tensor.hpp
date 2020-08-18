/**
 * @file tensor.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Tensor arithmetics and allocation implementation for CPU
 *
 * @copyright Copyright (c) 2020 UpStride
 */
#pragma once

#if __AVX512F__ || __AVX__
#include <immintrin.h>
#elif __SSE4_1__
#include <smmintrin.h>
#endif

#include "../backend.hpp"

namespace upstride {
template <>
struct TensorManipulations<device::CPU> {
    template <typename T>
    static void accumulateAdd(const Tensor<device::CPU, T>& input, Tensor<device::CPU, T>& output, const Shape& shape)  {
        int shapeNumel = shape.numel();
        T* outputPtr = output.getDataPtr();
        const T* inputPtr = input.getDataPtr();
        for (int i = 0; i < shapeNumel; ++i) {
            outputPtr[i] += inputPtr[i];
        }
    }
    static void accumulateAdd(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output, const Shape& shape); 
    static void accumulateAdd(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output, const Shape& shape); 

    template <typename T>
    static void accumulateSub(const Tensor<device::CPU, T>& input, Tensor<device::CPU, T>& output, const Shape& shape) {
        int shapeNumel = shape.numel();
        T* outputPtr = output.getDataPtr();
        const T* inputPtr = input.getDataPtr();
        for (int i = 0; i < shapeNumel; ++i) {
            outputPtr[i] -= inputPtr[i];
        }
    }
    static void accumulateSub(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output, const Shape& shape); 
    static void accumulateSub(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output, const Shape& shape);
    
    template <typename T>
    static inline void zero(Tensor<device::CPU, T>& output) {
        T* tensor = output.getDataPtr();
        memset(tensor, 0, output.getShape().numel() * sizeof(T));
    }
};

/**
 * @brief Implementation of allocated Tensor for CPU/oneDNN.
 * This tensor manages its own memory. Its instantiation/destruction implies memory allocation/disposition. Do not abuse with this, use with care!
 * @tparam T scalar datatype of tensor entries
 */
template <typename T>
class AllocatedTensor<device::CPU, T> : public Tensor<device::CPU, T> {
    AllocatedTensor(const Shape&, T*) = delete;  // deleting Tensor constructor allowing to wrap an external pointer
    using Tensor<device::CPU, T>::tensor;

   public:
    AllocatedTensor(const Shape& shape) : Tensor<device::CPU, T>(shape, nullptr) {
        tensor = (T*)malloc(shape.numel() * sizeof(T));
    }

    ~AllocatedTensor() {
        free(tensor);
    }
};
}  // namespace upstride