#pragma once
#include "utils.hpp"
#include "conv2d.hpp"
#include "dense.hpp"

namespace upstride {
    // upstride.hpp
    template <typename Device, typename T>
    void conv2DFwdTest(Device& device,
                       const Tensor<Device, const T>& inputTensor,
                       const Tensor<Device, const T>& filterTensor,
                       const Tensor<Device, const T>* biasTensor,
                       Tensor<Device, T>& outputTensor,
                       const Conv2DFwdDescriptor& descriptor) { }

    template <typename Device, typename T>
    void conv2DBwdTest(Device& device,
                       const Tensor<Device, const T>& inputTensor,
                       const Tensor<Device, const T>& filterTensor,
                       const Tensor<Device, const T>& gradTensor,
                       Tensor<Device, T>& filterGradTensor,
                       Tensor<Device, T>& inputGradTensor,
                       const Conv2DBwdDescriptor& descriptor) { }

    template <typename Device, typename T>
    void denseFwdTest(Device& device,
                       const Tensor<Device, const T>& inputTensor,
                       const Tensor<Device, const T>& filterTensor,
                       const Tensor<Device, const T>* biasTensor,
                       Tensor<Device, T>& outputTensor,
                       const DenseFwdDescriptor& descriptor) { }

    template <typename Device, typename T>
    void denseBwdTest(Device& device,
                       const Tensor<Device, const T>& inputTensor,
                       const Tensor<Device, const T>& filterTensor,
                       const Tensor<Device, const T>& gradTensor,
                       Tensor<Device, T>& filterGradTensor,
                       Tensor<Device, T>& inputGradTensor,
                       const DenseBwdDescriptor& descriptor) { }
}