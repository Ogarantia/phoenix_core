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
               const Tensor<Device, const T>& filterTensor,
               const Tensor<Device, const T>* biasTensor,
               Tensor<Device, T>& outputTensor,
               const Conv2DFwdDescriptor& descriptor)
{
    auto& op = device.template getConv2DFwdOperation<UpstrideConv2DFunctor<Device, T>>(descriptor);
    op(device, inputTensor, filterTensor, biasTensor, outputTensor, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
}

template <typename Device, typename T>
void conv2DBwd(Context& context,
               Device& device,
               const Tensor<Device, const T>& inputTensor,
               const Tensor<Device, const T>& filterTensor,
               const Tensor<Device, const T>& gradTensor,
               Tensor<Device, T>& filterGradTensor,
               Tensor<Device, T>& inputGradTensor,
               const Conv2DBwdDescriptor& descriptor)
{
    auto& op = device.template getConv2DBwdOperation<UpstrideConv2DGradFunctor<Device, T>>(descriptor);
    op(device, inputTensor, filterTensor, gradTensor, filterGradTensor, inputGradTensor, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
}

template <typename Device, typename T>
void denseFwd(Context& context,
              Device& device,
              const Tensor<Device, const T>& inputTensor,
              const Tensor<Device, const T>& filterTensor,
              const Tensor<Device, const T>* biasTensor,
              Tensor<Device, T>& outputTensor,
              const DenseFwdDescriptor& descriptor)
{
    auto& op = device.template getDenseFwdOperation<UpstrideDenseFunctor<Device, T>>(descriptor);
    op(device, inputTensor, filterTensor, biasTensor, outputTensor);
}

template <typename Device, typename T>
void denseBwd(Context& context,
              Device& device,
              const Tensor<Device, const T>& inputTensor,
              const Tensor<Device, const T>& filterTensor,
              const Tensor<Device, const T>& gradTensor,
              Tensor<Device, T>& filterGradTensor,
              Tensor<Device, T>& inputGradTensor,
              const DenseBwdDescriptor& descriptor)
{
    auto& op = device.template getDenseBwdOperation<UpstrideDenseGradFunctor<Device, T>>(descriptor);
    op(device, inputTensor, filterTensor, gradTensor, filterGradTensor, inputGradTensor);
}

}