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
    static void assignContents(Tensor<device::CUDA, T>& tensor, const std::vector<T>& contents) {
        if (tensor.getShape().numel() != contents.size())
            throw std::invalid_argument("Cannot assign tensor contents: size mismatch");
        cudnn::Context::raiseIfError(
            cudaMemcpyAsync(tensor.getDataPtr(), contents.data(), contents.size() * sizeof(T), cudaMemcpyHostToDevice, tensor.getDevice().stream()));
        cudnn::Context::raiseIfError(cudaStreamSynchronize(tensor.getDevice().stream()));
    }

    template <typename tensor_t, typename vector_t>
    static void getContents(const Tensor<device::CUDA, tensor_t>& tensor, std::vector<vector_t>& contents) {
        contents.resize(tensor.getShape().numel());
        static_assert(sizeof(tensor_t) == sizeof(vector_t), "Scalar datatype mismatch when copying a tensor content into a vector");
        cudnn::Context::raiseIfError(
            cudaMemcpyAsync(contents.data(), tensor.getDataPtr(), contents.size() * sizeof(vector_t), cudaMemcpyDeviceToHost, tensor.getDevice().stream()));
        cudnn::Context::raiseIfError(cudaStreamSynchronize(tensor.getDevice().stream()));
    }

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
    using Tensor<device::CUDA, T>::operator=;

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