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
#include "context.hpp"
#include "device.hpp"
#include "kernels.hpp"

namespace upstride {

template <>
struct TensorManipulations<device::CUDA> {
    template <typename T>
    static void accumulateAdd(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output) {
        cudnn::accumulateAdd(output.getDevice(), output.getDataPtr(), input.getDataPtr(), input.getShape().numel());
    }

    template <typename T>
    static void accumulateSub(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output) {
        cudnn::accumulateSub(output.getDevice(), output.getDataPtr(), input.getDataPtr(), input.getShape().numel());
    }

    template <typename T>
    static inline void zero(Tensor<device::CUDA, T>& output) {
        auto status = cudaMemsetAsync(output.getDataPtr(), 0, output.getShape().numel() * sizeof(T), output.getDevice().stream());
        if (status != cudaError::cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }

    template <typename T>
    static inline void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const T, 4>& inLeft, TemporaryTensor<device::CUDA, T>* outLeft,
                                                 const TensorSplit<device::CUDA, const T, 4>& inRight, TemporaryTensor<device::CUDA, T>* outRight) {
        cudnn::decomposeQuaternionInputs(inLeft, outLeft, inRight, outRight);
    }

    template <typename T>
    static inline void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const T, 4>& inGrad, TemporaryTensor<device::CUDA, T>* outGrad) {
        cudnn::decomposeQuaternionOutputGrad(inGrad, outGrad);
    }

    template <typename T>
    static inline void recomposeQuaternionOutput(TemporaryTensor<device::CUDA, T>* inLanes, TensorSplit<device::CUDA, T, 4>& outQuats) {
        cudnn::recomposeQuaternionOutput(inLanes, outQuats);
    }

    template <typename T>
    static inline void recomposeQuaternionInputsGrad(TemporaryTensor<device::CUDA, T>* inLeftGradLanes, TensorSplit<device::CUDA, T, 4>& outLeftGradQuats,
                                                     TemporaryTensor<device::CUDA, T>* inRightGradLanes, TensorSplit<device::CUDA, T, 4>& outRightGradQuats) {
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
    using Tensor<device::CUDA, T>::shape;
    using Tensor<device::CUDA, T>::device;
    int capacity;

   public:
    AllocatedTensor(device::CUDA& device, const Shape& shape) : Tensor<device::CUDA, T>(device, shape, nullptr), capacity(shape.numel()) {
        tensor = device.template malloc<T>(shape.numel() * sizeof(T));
    }

    ~AllocatedTensor() {
        device.free(tensor);
    }

    void reshape(const Shape& shape) {
        if (shape.numel() > capacity) {
            capacity = shape.numel();
            device.free(tensor);
            tensor = device.template malloc<T>(capacity * sizeof(T));
        }
        this->shape = shape;
    }
};
}  // namespace upstride