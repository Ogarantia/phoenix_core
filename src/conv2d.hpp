/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementations for different algebras
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include "backend/api.h"
#include "utils.hpp"

namespace upstride {

template <typename Device, typename T>
class UpstrideConv2DFunctor {
   private:
    ScalarConv2DFunctor<Device, T> convOp;  //!< scalar convolution operator to be used to implement other data types

   public:
    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    void configure(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation) {
        convOp.configure(dataFormat, stride, dilation);
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param kernelTensor      kernel tensor
     * @param outputTensor      Output tensor
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& kernelTensor,
                    Tensor<T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // fixme: this is scalar sconvolution operator
        convOp(inputTensor, kernelTensor, outputTensor, padBefore, padAfter, groups);
        // TODO: implement quaternion convolution right here
    }
};

template <typename Device, typename T>
class UpstrideConv2DGradFunctor {
   private:
    ScalarConv2DGradFunctor<Device, T> convOp;  //!< scalar convolution operator to be used to implement other data types

   public:
    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed as well
     */
    void configure(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) {
        convOp.configure(dataFormat, stride, dilation, requireInputGrad);
    }

    /**
     * @brief Executes the operation
     * @param inputTensor       forward input tensor
     * @param kernelTensor      forward input kernel tensor
     * @param gradTensor        gradient of the forward output tensor (dy)
     * @param kernelGradTensor  output: kernel gradient
     * @param inputGradTensor   output: input gradient
     * @param padBefore         number of zero samples to add to the input tensor on top/left
     * @param padAfter          number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& kernelTensor,
                    const Tensor<const T>& gradTensor,
                    Tensor<T>& kernelGradTensor,
                    Tensor<T>& inputGradTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // fixme: this is scalar sconvolution operator
        convOp(inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor, padBefore, padAfter, groups);
        // TODO: implement quaternion convolution right here
    }
};

}  // namespace upstride