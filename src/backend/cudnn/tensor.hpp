/**
 * @file tensor.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Tensor arithmetics and allocation implementation using CUDA
 *
 * @copyright Copyright (c) 2020 UpStride
 */
#pragma once

#include <cudnn.h>

#include "../backend.hpp"
#include "kernels.hpp"

namespace upstride {

template <>
struct TensorManipulations<device::CUDA> {
    template <typename T>
    static void accumulateAdd(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output) {
        cudnn::accumulateAdd(output.getDataPtr(), input.getDataPtr(), input.getShape().numel());
    }

    template <typename T>
    static void accumulateSub(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output) {
        cudnn::accumulateSub(output.getDataPtr(), input.getDataPtr(), input.getShape().numel());
    }

    template <typename T>
    static inline void zero(Tensor<device::CUDA, T>& output) {
        auto status = cudaMemset(output.getDataPtr(), 0, output.getShape().numel() * sizeof(T));
        if (status != cudaError::cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }

    template <typename T>
    static inline void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const T, 4>& inLeft, AllocatedTensor<device::CUDA, T>* outLeft[8],
                                                 const TensorSplit<device::CUDA, const T, 4>& inRight, AllocatedTensor<device::CUDA, T>* outRight[8]) {
        cudnn::decomposeQuaternionInputs(inLeft, outLeft, inRight, outRight);
    }

    template <typename T>
    static inline void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const T, 4>& inGrad, AllocatedTensor<device::CUDA, T>* outGrad[8]) {
        cudnn::decomposeQuaternionOutputGrad(inGrad, outGrad);
    }

    template <typename T>
    static inline void recomposeQuaternionOutput(AllocatedTensor<device::CUDA, T>* inLanes[8], TensorSplit<device::CUDA, T, 4>& outQuats) {
        cudnn::recomposeQuaternionOutput(inLanes, outQuats);
    }

    template <typename T>
    static inline void recomposeQuaternionInputsGrad(AllocatedTensor<device::CUDA, T>* inLeftGradLanes[8], TensorSplit<device::CUDA, T, 4>& outLeftGradQuats,
                                                     AllocatedTensor<device::CUDA, T>* inRightGradLanes[8], TensorSplit<device::CUDA, T, 4>& outRightGradQuats) {
        cudnn::recomposeQuaternionInputsGrad(inLeftGradLanes, outLeftGradQuats, inRightGradLanes, outRightGradQuats);
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
    using Tensor<device::CUDA, T>::tensor;

   public:
    AllocatedTensor(const Shape& shape) : Tensor<device::CUDA, T>(shape, nullptr) {
        auto status = cudaMalloc(&tensor, shape.numel() * sizeof(T));
        if (status != cudaError::cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }

    ~AllocatedTensor() {
        cudaFree(tensor);
    }
};
}  // namespace upstride