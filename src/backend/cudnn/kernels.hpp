/**
 * @file kernels.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Bunch of helpful CUDA kernels
 * 
 * @copyright Copyright (c) 2020 UpStride 
 */

#pragma once
#include "../tensor.hpp"
#include "../backend.hpp"

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

}  // namespace cudnn

}  // namespace upstride