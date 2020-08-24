/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementations for different algebras
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include "algebra_select_mixin.hpp"
#include "algebras.hpp"
#include "backend/api.h"
#include "thread_local_ptr.hpp"
#include "utils.hpp"

namespace upstride {

template <typename Device, typename T>
class UpstrideConv2DFunctor : public AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>> {
    using AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>>::proceedWithAlgebra;

   private:
    ThreadLocalPtr<ScalarConv2DFunctor<Device, T>> convOp;  //!< scalar convolution operator to be used to implement other data types
    Algebra algebra;
    DataFormat dataFormat;
    IntPair stride, dilation;

   public:
    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param algebra       Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    void configure(Algebra algebra, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation) {
        this->algebra = algebra;
        this->dataFormat = dataFormat;
        this->stride = stride;
        this->dilation = dilation;
    }

    /**
     * @brief Executes the convolution operation
     * This function may be called from multiple threads.
     * @param inputTensor       Input tensor
     * @param kernelTensor      kernel tensor
     * @param outputTensor      Output tensor
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void operator()(const Tensor<Device, const T>& inputTensor,
                    const Tensor<Device, const T>& kernelTensor,
                    Tensor<Device, T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // ensure the object exists within the current thread
        convOp(dataFormat, stride, dilation);

        convOp->configure(
            inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            kernelTensor.getShape().slice(-4),
            outputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            padBefore,
            padAfter,
            groups);

        proceedWithAlgebra(algebra, inputTensor, kernelTensor, outputTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            Tensor<Device, T>& outputTensor) {
        using CliffordProductSpec = CliffordProductSpec<algebra>;

        // split tensors along blades
        TensorSplit<Device, const T, CliffordProductSpec::DIMS>
            input(inputTensor),
            kernel(kernelTensor, kernelTensor.getShape().getSize() == 4);
        TensorSplit<Device, T, CliffordProductSpec::DIMS> output(outputTensor);

        // allocate a temporary buffer
        AllocatedTensor<Device, T> buffer(output.shape());

        // compute the Clifford product
        BinaryOperation<CliffordProductSpec>::product(
            [this, &input, &kernel, &output](int left, int right, int dim) {
                (*convOp)(input[left], kernel[right], output[dim]);
            },

            [this, &input, &kernel, &buffer](int left, int right, int) {
                (*convOp)(input[left], kernel[right], buffer);
            },

            [this, &output, &buffer](int dim, int, bool positive) {
                if (positive)
                    output[dim] += buffer;
                else
                    output[dim] -= buffer;
            });
    }
};

template <typename Device, typename T>
class UpstrideConv2DGradFunctor : public AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>> {
    using AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>>::proceedWithAlgebra;

   private:
    ThreadLocalPtr<ScalarConv2DGradFunctor<Device, T>> convOp;  //!< scalar convolution operator to be used to implement other data types
    Algebra algebra;
    DataFormat dataFormat;
    IntPair stride, dilation;
    bool requireInputGrad;

   public:
    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param algebra       Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed as well
     */
    void configure(Algebra algebra, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) {
        this->algebra = algebra;
        this->dataFormat = dataFormat;
        this->stride = stride;
        this->dilation = dilation;
        this->requireInputGrad = requireInputGrad;
    }

    /**
     * @brief Executes the operation
     * This function may be called from multiple threads.
     * @param inputTensor       forward input tensor
     * @param kernelTensor      forward input kernel tensor
     * @param gradTensor        gradient of the forward output tensor (dy)
     * @param kernelGradTensor  output: kernel gradient
     * @param inputGradTensor   output: input gradient
     * @param padBefore         number of zero samples to add to the input tensor on top/left
     * @param padAfter          number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void operator()(const Tensor<Device, const T>& inputTensor,
                    const Tensor<Device, const T>& kernelTensor,
                    const Tensor<Device, const T>& gradTensor,
                    Tensor<Device, T>& kernelGradTensor,
                    Tensor<Device, T>& inputGradTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // ensure the object exists within the current thread
        convOp(dataFormat, stride, dilation, requireInputGrad);

        convOp->configure(
            inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            kernelTensor.getShape().slice(-4),
            gradTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            padBefore,
            padAfter,
            groups);

        proceedWithAlgebra(algebra, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>& gradTensor,
                            Tensor<Device, T>& kernelGradTensor,
                            Tensor<Device, T>& inputGradTensor) {
        using CliffordProductSpec = CliffordProductSpec<algebra>;

        // split tensors along blades
        TensorSplit<Device, const T, CliffordProductSpec::DIMS>
            input(inputTensor),
            kernel(kernelTensor, kernelTensor.getShape().getSize() == 4),
            grad(gradTensor);
        TensorSplit<Device, T, CliffordProductSpec::DIMS> outputKernel(kernelGradTensor);
        TensorSplit<Device, T, CliffordProductSpec::DIMS> outputInput(inputGradTensor);

        // allocate a temporary buffer
        AllocatedTensor<Device, T> bufferKernel(outputKernel.shape());
        AllocatedTensor<Device, T> bufferInput(outputInput.shape());

        // compute the Clifford product
        BinaryOperation<CliffordProductSpec>::product(
            [this, &input, &kernel, &grad, &outputKernel, &outputInput](int left, int right, int dim) {
                (*convOp)(input[left], kernel[right], grad[dim], outputKernel[dim], outputInput[dim]);
            },

            [this, &input, &kernel, &grad, &bufferKernel, &bufferInput](int left, int right, int dim) {
                (*convOp)(input[left], kernel[right], grad[dim], bufferKernel, bufferInput);
            },

            [this, &outputKernel, &outputInput, &bufferKernel, &bufferInput](int dim, int, bool positive) {
                if (positive) {
                    outputKernel[dim] += bufferKernel;
                    outputInput[dim] += bufferInput;
                } else {
                    outputKernel[dim] -= bufferKernel;
                    outputInput[dim] -= bufferInput;
                }
            });
    }
};

}  // namespace upstride