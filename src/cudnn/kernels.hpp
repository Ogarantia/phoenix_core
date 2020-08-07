/**
 * @file kernels.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Bunch of helpful CUDA kernels
 * 
 * @copyright Copyright (c) 2020 UpStride 
 */
#pragma once
#include "../utils.hpp"

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
void crop(const Tensor<const T>& input, Tensor<T>& output, DataFormat dataFormat, const IntPair& offset);

template <>
void crop(const Tensor<const float>& input, Tensor<float>& output, DataFormat dataFormat, const IntPair& offset);

}  // namespace cudnn
}  // namespace upstride