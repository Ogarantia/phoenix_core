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
#include "device.hpp"

namespace upstride {
template <>
struct TensorManipulations<device::CPU> {
    template <typename T>
    static void accumulateAdd(const Tensor<device::CPU, T>& input, Tensor<device::CPU, T>& output) {
        const int shapeNumel = input.getShape().numel();
        T* outputPtr = output.getDataPtr();
        const T* inputPtr = input.getDataPtr();
        for (int i = 0; i < shapeNumel; ++i) {
            outputPtr[i] += inputPtr[i];
        }
    }
    static void accumulateAdd(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output);
    static void accumulateAdd(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output);

    template <typename T>
    static void accumulateSub(const Tensor<device::CPU, T>& input, Tensor<device::CPU, T>& output) {
        const int shapeNumel = input.getShape().numel();
        T* outputPtr = output.getDataPtr();
        const T* inputPtr = input.getDataPtr();
        for (int i = 0; i < shapeNumel; ++i) {
            outputPtr[i] -= inputPtr[i];
        }
    }
    static void accumulateSub(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output);
    static void accumulateSub(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output);

    template <typename T>
    static inline void zero(Tensor<device::CPU, T>& output) {
        T* tensor = output.getDataPtr();
        memset(tensor, 0, output.getShape().numel() * sizeof(T));
    }

    template <typename T>
    static inline void decomposeQuaternionInputs(
        const TensorSplit<device::CPU, const T, 4>& inLeft, AllocatedTensor<device::CPU, T>* outLeft[8],
        const TensorSplit<device::CPU, const T, 4>& inRight, AllocatedTensor<device::CPU, T>* outRight[8]) {
        // compute left operand decomposition
        {
            const int length = inLeft.shape().numel();
            const T* inPtr[] = {inLeft[0].getDataPtr(), inLeft[1].getDataPtr(), inLeft[2].getDataPtr(), inLeft[3].getDataPtr()};
            T* outPtr[] = {outLeft[0]->getDataPtr(), outLeft[1]->getDataPtr(), outLeft[2]->getDataPtr(), outLeft[3]->getDataPtr(),
                           outLeft[4]->getDataPtr(), outLeft[5]->getDataPtr(), outLeft[6]->getDataPtr(), outLeft[7]->getDataPtr()};
            for (int i = 0; i < length; ++i) {
                outPtr[0][i] = inPtr[3][i] + inPtr[1][i];
                outPtr[1][i] = inPtr[0][i] - inPtr[2][i];
                outPtr[2][i] = inPtr[0][i] + inPtr[2][i];
                outPtr[3][i] = inPtr[3][i] - inPtr[1][i];
                outPtr[4][i] = inPtr[3][i] - inPtr[2][i];
                outPtr[5][i] = inPtr[1][i] + inPtr[0][i];
                outPtr[6][i] = inPtr[0][i] - inPtr[1][i];
                outPtr[7][i] = inPtr[3][i] + inPtr[2][i];
            }
        }

        // compute right operand decomposition
        {
            const int length = inRight.shape().numel();
            const T* inPtr[] = {inRight[0].getDataPtr(), inRight[1].getDataPtr(), inRight[2].getDataPtr(), inRight[3].getDataPtr()};
            T* outPtr[] = {outRight[0]->getDataPtr(), outRight[1]->getDataPtr(), outRight[2]->getDataPtr(), outRight[3]->getDataPtr(),
                           outRight[4]->getDataPtr(), outRight[5]->getDataPtr(), outRight[6]->getDataPtr(), outRight[7]->getDataPtr()};
            for (int i = 0; i < length; ++i) {
                outPtr[0][i] = inPtr[1][i] + inPtr[2][i];
                outPtr[1][i] = inPtr[0][i] + inPtr[3][i];
                outPtr[2][i] = inPtr[0][i] - inPtr[3][i];
                outPtr[3][i] = inPtr[1][i] - inPtr[2][i];
                outPtr[4][i] = inPtr[2][i] - inPtr[3][i];
                outPtr[5][i] = inPtr[1][i] + inPtr[0][i];
                outPtr[6][i] = inPtr[2][i] + inPtr[3][i];
                outPtr[7][i] = inPtr[0][i] - inPtr[1][i];
            }
        }
    }

    template <typename T>
    static inline void decomposeQuaternionOutputGrad(const TensorSplit<device::CPU, const T, 4>& inGrad, AllocatedTensor<device::CPU, T>* outGrad[8]) {
        const int length = inGrad.shape().numel();
        const T* inPtr[] = {inGrad[0].getDataPtr(), inGrad[1].getDataPtr(), inGrad[2].getDataPtr(), inGrad[3].getDataPtr()};
        T* outPtr[] = {outGrad[0]->getDataPtr(), outGrad[1]->getDataPtr(), outGrad[2]->getDataPtr(), outGrad[3]->getDataPtr(),
                       outGrad[4]->getDataPtr(), outGrad[5]->getDataPtr(), outGrad[6]->getDataPtr(), outGrad[7]->getDataPtr()};
        for (int i = 0; i < length; ++i) {
            const T t1 = inPtr[0][i] + inPtr[1][i];
            const T t3 = inPtr[0][i] - inPtr[1][i];
            const T t2 = inPtr[2][i] + inPtr[3][i];
            const T t4 = inPtr[2][i] - inPtr[3][i];
            outPtr[0][i] = (T).5 * (t2 - t1);
            outPtr[1][i] = (T).5 * (t3 - t4);
            outPtr[2][i] = (T).5 * (t3 + t4);
            outPtr[3][i] = (T).5 * (t1 + t2);
            outPtr[4][i] = inPtr[0][i];
            outPtr[5][i] = inPtr[1][i];
            outPtr[6][i] = inPtr[2][i];
            outPtr[7][i] = inPtr[3][i];
        }
    }

    template <typename T>
    static inline void recomposeQuaternionOutput(AllocatedTensor<device::CPU, T>* inLanes[8], TensorSplit<device::CPU, T, 4>& outQuats) {
        const int length = outQuats.shape().numel();
        const T* inPtr[] = {inLanes[0]->getDataPtr(), inLanes[1]->getDataPtr(), inLanes[2]->getDataPtr(), inLanes[3]->getDataPtr(),
                            inLanes[4]->getDataPtr(), inLanes[5]->getDataPtr(), inLanes[6]->getDataPtr(), inLanes[7]->getDataPtr()};
        T* outPtr[] = {outQuats[0].getDataPtr(), outQuats[1].getDataPtr(), outQuats[2].getDataPtr(), outQuats[3].getDataPtr()};
        for (int i = 0; i < length; ++i) {
            const T a2 = inPtr[0][i] + inPtr[1][i] + inPtr[2][i];
            const T a5 = 0.5 * (a2 + inPtr[3][i]);
            outPtr[0][i] = a5 - inPtr[0][i] + inPtr[4][i];
            outPtr[1][i] = a5 - a2 + inPtr[5][i];
            outPtr[2][i] = a5 - inPtr[1][i] + inPtr[6][i];
            outPtr[3][i] = a5 - inPtr[2][i] + inPtr[7][i];
        }
    }

    template <typename T>
    static inline void recomposeQuaternionInputsGrad(AllocatedTensor<device::CPU, T>* inLeftGradLanes[8], TensorSplit<device::CPU, T, 4>& outLeftGradQuats,
                                                     AllocatedTensor<device::CPU, T>* inRightGradLanes[8], TensorSplit<device::CPU, T, 4>& outRightGradQuats) {
        // compute left operand gradient
        {
            const int length = outLeftGradQuats.shape().numel();
            const T* inPtr[] = {inLeftGradLanes[0]->getDataPtr(), inLeftGradLanes[1]->getDataPtr(), inLeftGradLanes[2]->getDataPtr(), inLeftGradLanes[3]->getDataPtr(),
                                inLeftGradLanes[4]->getDataPtr(), inLeftGradLanes[5]->getDataPtr(), inLeftGradLanes[6]->getDataPtr(), inLeftGradLanes[7]->getDataPtr()};
            T* outPtr[] = {outLeftGradQuats[0].getDataPtr(), outLeftGradQuats[1].getDataPtr(), outLeftGradQuats[2].getDataPtr(), outLeftGradQuats[3].getDataPtr()};
            for (int i = 0; i < length; ++i) {
                outPtr[0][i] = inPtr[1][i] + inPtr[2][i] + inPtr[5][i] + inPtr[6][i];
                outPtr[1][i] = inPtr[0][i] - inPtr[3][i] + inPtr[5][i] - inPtr[6][i];
                outPtr[2][i] = inPtr[2][i] - inPtr[1][i] - inPtr[4][i] + inPtr[7][i];
                outPtr[3][i] = inPtr[0][i] + inPtr[3][i] + inPtr[4][i] + inPtr[7][i];
            }
        }

        // compute right operand gradient
        {
            const int length = outRightGradQuats.shape().numel();
            const T* inPtr[] = {inRightGradLanes[0]->getDataPtr(), inRightGradLanes[1]->getDataPtr(), inRightGradLanes[2]->getDataPtr(), inRightGradLanes[3]->getDataPtr(),
                                inRightGradLanes[4]->getDataPtr(), inRightGradLanes[5]->getDataPtr(), inRightGradLanes[6]->getDataPtr(), inRightGradLanes[7]->getDataPtr()};
            T* outPtr[] = {outRightGradQuats[0].getDataPtr(), outRightGradQuats[1].getDataPtr(), outRightGradQuats[2].getDataPtr(), outRightGradQuats[3].getDataPtr()};
            for (int i = 0; i < length; ++i) {
                outPtr[0][i] = inPtr[1][i] + inPtr[2][i] + inPtr[5][i] + inPtr[7][i];
                outPtr[1][i] = inPtr[0][i] + inPtr[3][i] + inPtr[5][i] - inPtr[7][i];
                outPtr[2][i] = inPtr[0][i] - inPtr[3][i] + inPtr[4][i] + inPtr[6][i];
                outPtr[3][i] = inPtr[1][i] - inPtr[2][i] - inPtr[4][i] + inPtr[6][i];
            }
        }
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
    using Tensor<device::CPU, T>::shape;

   public:
    AllocatedTensor(const device::CPU& device, const Shape& shape) : Tensor<device::CPU, T>(device, shape, nullptr) {
        tensor = (T*)malloc(shape.numel() * sizeof(T));
    }

    AllocatedTensor(const device::CPU& device) : Tensor<device::CPU, T>(device, Shape(), nullptr) {}

    ~AllocatedTensor() {
        free(tensor);
    }

    void reshape(const Shape& shape) {
        if (this->shape != shape) {
            free(tensor);
            tensor = (T*)malloc(shape.numel() * sizeof(T));
            this->shape = shape;
        }
    }
};
}  // namespace upstride