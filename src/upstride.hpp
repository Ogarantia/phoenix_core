/**
 * @file upstride.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief UpStride API
 * Include this file to get the whole package at once
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include "utils.hpp"
#include "conv2d.hpp"
#include "dense.hpp"

namespace upstride {

template <typename Device, typename T>
void conv2DFwd(Context& context,
               Device& device,
               const Tensor<Device, const T>& inputTensor,
               const Tensor<Device, const T>& kernelTensor,
               const Tensor<Device, const T>* biasTensor,
               Tensor<Device, T>& outputTensor,
               const Conv2DDescriptor& descriptor) {

    UpstrideConv2DFunctor<Device, T> functor(context, descriptor);

    functor(device, inputTensor, kernelTensor, biasTensor, outputTensor, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
}

}