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
#include "deferred_allocator.hpp"
#include "thread_local_ptr.hpp"
#include "utils.hpp"

namespace upstride {

/**
 * @brief Specifies memory layout for a convolution kernel
 *   OIHW for real algebra
 *   nOIHW for a non-real algebra containinb n-dimensional multivectors.
 */
class Conv2DKernelLayout {
   public:
    /**
     * @brief Returns dimension number containing the number of output channels in the convolution kernel for a specific algebra.
     */
    static inline int numOutputChannelsDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 0 : 1;
    }

    /**
     * @brief Returns dimension number containing the number of input channels in the convolution kernel for a specific algebra.
     */
    static inline int numInputChannelsDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 1 : 2;
    }

    /**
     * @brief Returns dimension number containing the height of the convolution kernel for a specific algebra.
     */
    static inline int heightDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 2 : 3;
    }

    /**
     * @brief Returns dimension number containing the width of the convolution kernel for a specific algebra.
     */
    static inline int widthDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 3 : 4;
    }
};

template <typename Device, typename T>
class UpstrideConv2DFunctor : public AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>> {
    using AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>>::proceedWithAlgebra;

   private:
    ThreadLocalPtr<ScalarConv2DFunctor<Device, T>> convOp;  //!< scalar convolution operator to be used to implement other data types
    Algebra algebra;
    DataFormat dataFormat;
    IntPair stride, dilation;
    DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], outputLanes[8];  //!< deferred allocators for the factorized quaternion implementation
    DeferredAllocator<Device, T> buffer;                                         //!< deferred allocator for an intermediate buffer for the default implementation

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
     * @param kernelTensor      Kernel tensor
     * @param biasTensor        Pointer to bias tensor; may be null
     * @param outputTensor      Output tensor
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void operator()(const Tensor<Device, const T>& inputTensor,
                    const Tensor<Device, const T>& kernelTensor,
                    const Tensor<Device, const T>* biasTensor,
                    Tensor<Device, T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // ensure the object exists within the current thread
        convOp(dataFormat, stride, dilation, biasTensor != nullptr);

        convOp->configure(
            inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            kernelTensor.getShape().slice(-4),
            biasTensor ? Shape{biasTensor->getShape().numel() / MULTIVECTOR_DIM[algebra]} : Shape(),
            outputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            padBefore,
            padAfter,
            groups);

        proceedWithAlgebra(algebra, inputTensor, kernelTensor, biasTensor, outputTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>* biasTensor,
                            Tensor<Device, T>& outputTensor) {
        if (algebra == Algebra::QUATERNION && Context::preferSpeedToMemory()) {
            // split tensors along blades
            const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
            TensorSplit<Device, T, 4> output(outputTensor);

            // allocate bias tensor split if the bias tensor is provided
            TensorSplit<Device, const T, 4>* bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

            // get temporary buffers
            AllocatedTensor<Device, T>*inputLanes[8], *kernelLanes[8], *outputLanes[8];
            for (int i = 0; i < 8; ++i) {
                inputLanes[i] = &this->inputLanes[i].get(inputTensor.getDevice(), input.shape());
                kernelLanes[i] = &this->kernelLanes[i].get(kernelTensor.getDevice(), kernel.shape());
                outputLanes[i] = &this->outputLanes[i].get(outputTensor.getDevice(), output.shape());
            }

            // decompose - compute - recompose
            TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
            for (int i = 0; i < 4; ++i)
                (*convOp)(*inputLanes[i], *kernelLanes[i], nullptr, *outputLanes[i]);
            for (int i = 4; i < 8; ++i)
                // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                // Adding bias there then!
                (*convOp)(*inputLanes[i], *kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, *outputLanes[i]);

            TensorManipulations<Device>::recomposeQuaternionOutput(outputLanes, output);

            // free bias
            delete bias;
        }

        else {
            using CliffordProductSpec = CliffordProductSpec<algebra>;

            // split tensors along blades
            const TensorSplit<Device, const T, CliffordProductSpec::DIMS>
                input(inputTensor),
                kernel(kernelTensor, kernelTensor.getShape().getSize() == 4);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> output(outputTensor);

            // allocate bias tensor split if the bias tensor is provided
            TensorSplit<Device, const T, CliffordProductSpec::DIMS>* bias = biasTensor ? new TensorSplit<Device, const T, CliffordProductSpec::DIMS>(*biasTensor) : nullptr;

            // allocate a temporary buffer
            AllocatedTensor<Device, T>& buffer(this->buffer.get(outputTensor.getDevice(), output.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::product(
                [this, &input, &kernel, bias, &output](int left, int right, int dim) {
                    (*convOp)(input[left], kernel[right], bias ? &(*bias)[dim] : nullptr, output[dim]);
                },

                [this, &input, &kernel, &output, &buffer](int left, int right, int dim,  bool positive) {
                    (*convOp)(input[left], kernel[right], nullptr, buffer);
                    if (positive)
                        output[dim] += buffer;
                    else
                        output[dim] -= buffer;
                });

            // free bias
            delete bias;
        }
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
    DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], gradLanes[8], kernelGradLanes[8], inputGradLanes[8];  //!< deferred allocators for the factorized quaternion implementation
    DeferredAllocator<Device, T> bufferInput, bufferKernel;                                                           //!< deferred allocator for an intermediate buffer for the default implementation

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
        if (algebra == Algebra::QUATERNION && Context::preferSpeedToMemory()) {
            // split tensors along blades
            const TensorSplit<Device, const T, 4>
                input(inputTensor),
                kernel(kernelTensor, false),
                grad(gradTensor);
            TensorSplit<Device, T, 4> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

            // get temporary buffers
            AllocatedTensor<Device, T>*inputLanes[8], *kernelLanes[8], *gradLanes[8], *kernelGradLanes[8], *inputGradLanes[8];
            for (int i = 0; i < 8; ++i) {
                inputLanes[i] = &this->inputLanes[i].get(inputTensor.getDevice(), input.shape());
                kernelLanes[i] = &this->kernelLanes[i].get(kernelTensor.getDevice(), kernel.shape());
                gradLanes[i] = &this->gradLanes[i].get(gradTensor.getDevice(), grad.shape());
                kernelGradLanes[i] = &this->kernelGradLanes[i].get(kernelGradTensor.getDevice(), kernelGrad.shape());
                inputGradLanes[i] = &this->inputGradLanes[i].get(inputGradTensor.getDevice(), inputGrad.shape());
            }

            // decompose - compute - recompose
            TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
            TensorManipulations<Device>::decomposeQuaternionOutputGrad(grad, gradLanes);

            for (int i = 0; i < 8; ++i)
                (*convOp)(*inputLanes[i], *kernelLanes[i], *gradLanes[i], *kernelGradLanes[i], *inputGradLanes[i]);

            TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes, inputGrad, kernelGradLanes, kernelGrad);
        }

        else {
            using CliffordProductSpec = CliffordProductSpec<algebra>;

            // split tensors along blades
            const TensorSplit<Device, const T, CliffordProductSpec::DIMS>
                input(inputTensor),
                kernel(kernelTensor, kernelTensor.getShape().getSize() == 4),
                grad(gradTensor);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

            // allocate a temporary buffer
            AllocatedTensor<Device, T>& bufferKernel(this->bufferKernel.get(kernelGradTensor.getDevice(), kernelGrad.shape()));
            AllocatedTensor<Device, T>& bufferInput(this->bufferInput.get(inputGradTensor.getDevice(), inputGrad.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::productBackprop(
                [this, &input, &kernel, &grad, &kernelGrad, &inputGrad](int left, int right, int dim) {
                    (*convOp)(input[left], kernel[right], grad[dim], kernelGrad[right], inputGrad[left]);
                },

                [this, &input, &kernel, &grad, &kernelGrad, &inputGrad, &bufferKernel, &bufferInput](int left, int right, int dim, bool positive) {
                    (*convOp)(input[left], kernel[right], grad[dim], bufferKernel, bufferInput);
                    if (positive) {
                        kernelGrad[right] += bufferKernel;
                        inputGrad[left] += bufferInput;
                    } else {
                        kernelGrad[right] -= bufferKernel;
                        inputGrad[left] -= bufferInput;
                    }
                });
        }
    }
};

}  // namespace upstride