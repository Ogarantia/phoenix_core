/**
 * @file kernels.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Bunch of helpful CUDA kernels
 * 
 * @copyright Copyright (c) 2020 UpStride 
 */

#pragma once
#include "../backend.hpp"
#include "../tensor.hpp"

/**
 * @brief Rounding up integer division
 * @param n nominator
 * @param d denominator
 * @return closest integer greater than n/d
 */
static inline int ceili(int n, int d) {
    return (n + d - 1) / d;
}

namespace upstride {
namespace cudnn {

/**
 * @brief Crops a tensor along W and H dimensions
 *
 * @tparam  tensor datatype
 * @param input         input tensor
 * @param output        output tensor
 * @param dataFormat    input and output data format
 */
template <typename T>
extern void crop(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset);

/**
 * @brief Insert a tensor into another bigger tensor with a potential offset along the spatial dimensions H and W
 *
 * @tparam T the tensor datatype
 * @param input         input tensor
 * @param output        output tensor
 * @param dataFormat    input and output tensors data format
 * @param offset        position of the topleft corner of the input inside the output
 */
template <typename T>
extern void insert(const Tensor<device::CUDA, const T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset);

template <typename T>
extern void accumulateAdd(T* accumulator, const T* term, int length);

template <typename T>
extern void accumulateSub(T* accumulator, const T* term, int length);

template <typename T>
extern void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const T, 4>& inLeft, AllocatedTensor<device::CUDA, T>* outLeft[8],
                                      const TensorSplit<device::CUDA, const T, 4>& inRight, AllocatedTensor<device::CUDA, T>* outRight[8]);

template <typename T>
extern void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const T, 4>& inGrad, AllocatedTensor<device::CUDA, T>* outGrad[8]);

template <typename T>
extern void recomposeQuaternionOutput(AllocatedTensor<device::CUDA, T>* inLanes[8], TensorSplit<device::CUDA, T, 4>& outQuats);

template <typename T>
extern void recomposeQuaternionInputsGrad(AllocatedTensor<device::CUDA, T>* inLeftGradLanes[8], TensorSplit<device::CUDA, T, 4>& outLeftGradQuats,
                                          AllocatedTensor<device::CUDA, T>* inRightGradLanes[8], TensorSplit<device::CUDA, T, 4>& outRightGradQuats);

}  // namespace cudnn

}  // namespace upstride