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
void conv2DFwd(Device& device,
               const Tensor<Device, const T>& inputTensor,
               const Tensor<Device, const T>& filterTensor,
               const Tensor<Device, const T>* biasTensor,
               Tensor<Device, T>& outputTensor,
               const Conv2DFwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(device.getAccessControl());
    auto& op = device.template getConv2DFwdOperation<Device, UpstrideConv2DFunctor<Device, T>>(descriptor);
    op(inputTensor, filterTensor, biasTensor, outputTensor, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
}

template <typename Device, typename T>
void conv2DBwd(Device& device,
               const Tensor<Device, const T>& inputTensor,
               const Tensor<Device, const T>& filterTensor,
               const Tensor<Device, const T>& gradTensor,
               Tensor<Device, T>& filterGradTensor,
               Tensor<Device, T>& inputGradTensor,
               const Conv2DBwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(device.getAccessControl());
    auto& op = device.template getConv2DBwdOperation<Device, UpstrideConv2DGradFunctor<Device, T>>(descriptor);
    op(inputTensor, filterTensor, gradTensor, filterGradTensor, inputGradTensor, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
}

template <typename Device, typename T>
void denseFwd(Device& device,
              const Tensor<Device, const T>& inputTensor,
              const Tensor<Device, const T>& filterTensor,
              const Tensor<Device, const T>* biasTensor,
              Tensor<Device, T>& outputTensor,
              const DenseFwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(device.getAccessControl());
    auto& op = device.template getDenseFwdOperation<Device, UpstrideDenseFunctor<Device, T>>(descriptor);
    op(inputTensor, filterTensor, biasTensor, outputTensor);
}

template <typename Device, typename T>
void denseBwd(Device& device,
              const Tensor<Device, const T>& inputTensor,
              const Tensor<Device, const T>& filterTensor,
              const Tensor<Device, const T>& gradTensor,
              Tensor<Device, T>& filterGradTensor,
              Tensor<Device, T>& inputGradTensor,
              const DenseBwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(device.getAccessControl());
    auto& op = device.template getDenseBwdOperation<Device, UpstrideDenseGradFunctor<Device, T>>(descriptor);
    op(inputTensor, filterTensor, gradTensor, filterGradTensor, inputGradTensor);
}

}