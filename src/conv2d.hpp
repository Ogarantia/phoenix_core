/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementations for different algebras
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <mutex>

#include "algebra_select_mixin.hpp"
#include "algebras.hpp"
#include "backend/api.h"
#include "backend/conv2d_descriptor.hpp"
#include "backend/tensor.hpp"
#include "backend/operation.hpp"
#include "deferred_allocator.hpp"
#include "backend/temporary_tensor.hpp"
#include "utils.hpp"

namespace upstride {

template <typename Device, typename T>
class UpstrideConv2DFunctor : public AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>>, public Operation {
    using AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>>::proceedWithAlgebra;

   private:
    Device& device;                             //!< the device instance the operation is attached to
    const Algebra algebra;
    ScalarConv2DFunctor<Device, T> convOp;      //!< scalar convolution operator to be used to implement other data types
    cuda::QuatKernelPointwiseConvForwardFunctor<Device, T> quatKernelOp;       //!< custom kernels operator for quaternionic pointwise convolution
    DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], outputLanes[8];  //!< deferred allocators for the factorized quaternion implementation
    DeferredAllocator<Device, T> buffer;                                         //!< deferred allocator for an intermediate buffer for the default implementation
    const bool realValuedInput;                 //!< if `true`, the input tensor is real-valued (contains the real part only)
    std::mutex access;

   public:
    /**
     * @brief Instantiates convolution operator.
     * @param device        A device instance
     * @param descriptor    Operation descriptor
     */
    UpstrideConv2DFunctor(Device& device, const Conv2DFwdDescriptor& descriptor):
        device(device),
        algebra(descriptor.getAlgebra()),
        convOp(device.getContext(), descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation(), descriptor.isBiasUsed()),
        quatKernelOp(device.getContext(), algebra, descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation()),
        realValuedInput(descriptor.isRealValuedInput())
    {}

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
        // Sometimes TF sends us an empty tensor, cudnn does not digest this well
        if (inputTensor.getShape().empty())
            return;

        // lock access to convOp
        std::lock_guard<std::mutex> lock(access);

        convOp.configure(
            device,
            realValuedInput ? inputTensor.getShape() : inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            kernelTensor.getShape().slice(-4),
            biasTensor ? Shape{biasTensor->getShape().numel() / MULTIVECTOR_DIM[algebra]} : Shape(),
            outputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            padBefore,
            padAfter,
            groups);

        if (algebra == Algebra::QUATERNION && !realValuedInput) {
            quatKernelOp.configure(
                device,
                inputTensor.getShape(),
                kernelTensor.getShape(),
                padBefore,
                padAfter,
                groups);
        }

        proceedWithAlgebra(algebra, inputTensor, kernelTensor, biasTensor, outputTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>* biasTensor,
                            Tensor<Device, T>& outputTensor) {
        MemoryRequest memory(device, *this);

        // run custom convolution kernels for quaternions if possible
        if (quatKernelOp.canRun()) {
            quatKernelOp(device, inputTensor, kernelTensor, biasTensor, outputTensor);
        }

        // real-valued input
        else if (realValuedInput && algebra != Algebra::REAL) {
            using CliffordProductSpec = CliffordProductSpec<algebra>;

            // make sure there is no bias
            if (biasTensor)
                throw std::runtime_error("Bias addition is not supported for type0 inputs");

            // split tensors along blades
            const TensorSplit<Device, const T, CliffordProductSpec::DIMS> kernel(kernelTensor, false);
            const auto& input(inputTensor);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> output(outputTensor);

            // get a temporary buffer
            TemporaryTensor<Device, T> buffer(device, memory, output.shape());

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();

            // prepare buffer
            buffer.prepare();

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::product(
                [this, &input, &kernel, &output](int left, int right, int dim) {
                    if (left == 0)
                        convOp(input, kernel[right], nullptr, output[dim]);
                    // no else branch: input is zero if left > 0, bias is zero as well
                },

                [this, &input, &kernel, &output, &buffer](int left, int right, int dim, bool positive) {
                    if (left == 0) {
                        convOp(input, kernel[right], nullptr, buffer);
                        if (positive)
                            output[dim] += buffer;
                        else
                            output[dim] -= buffer;
                    }
                    // no else branch: input is zero if left > 0, bias is zero as well
                });
        }

        // factorized quaternion convolution fallback
        else if (algebra == Algebra::QUATERNION && device.getContext().preferSpeedToMemory()) {
            // split tensors along blades
            const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
            TensorSplit<Device, T, 4> output(outputTensor);

            // allocate bias tensor split if the bias tensor is provided
            TensorSplit<Device, const T, 4>* bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

            // get temporary buffers
            std::array<TemporaryTensor<Device, T>, 8> inputLanes{ {
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> kernelLanes{ {
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> outputLanes{ {
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() },
                { device, memory, output.shape() }
            } };

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();

            for (int i = 0; i < 8; ++i) {
                inputLanes[i].prepare();
                kernelLanes[i].prepare();
                outputLanes[i].prepare();
            }

            // decompose - compute - recompose
            TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes.data(), kernel, kernelLanes.data());
            for (int i = 0; i < 4; ++i)
                convOp(inputLanes[i], kernelLanes[i], nullptr, outputLanes[i]);
            for (int i = 4; i < 8; ++i)
                // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                // Adding bias there then!
                convOp(inputLanes[i], kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, outputLanes[i]);

            TensorManipulations<Device>::recomposeQuaternionOutput(outputLanes.data(), output);

            // free bias
            delete bias;
        }

        // generic implementation
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
            TemporaryTensor<Device, T> buffer(device, memory, output.shape());

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();
            buffer.prepare();

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::product(
                [this, &memory, &input, &kernel, bias, &output](int left, int right, int dim) {
                    convOp(input[left], kernel[right], bias ? &(*bias)[dim] : nullptr, output[dim]);
                },

                [this, &memory, &input, &kernel, &output, &buffer](int left, int right, int dim,  bool positive) {
                    convOp(input[left], kernel[right], nullptr, buffer);
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
class UpstrideConv2DGradFunctor : public AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>>, public Operation {
    using AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>>::proceedWithAlgebra;

   private:
    Device& device;                                             //!< the device instance the operation is attached to
    const Algebra algebra;
    ScalarConv2DGradFunctor<Device, T> convOp;                  //!< scalar convolution operator to be used to implement other data types
    cuda::QuatKernelPointwiseConvBackwardFunctor<Device, T> quatKernelOp;              //!< custom kernels operator for quaternionic pointwise convolution
    DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], gradLanes[8], kernelGradLanes[8], inputGradLanes[8];  //!< deferred allocators for the factorized quaternion implementation
    DeferredAllocator<Device, T> bufferInput, bufferKernel;                                                           //!< deferred allocator for an intermediate buffer for the default implementation
    const bool requireInputGrad;                                //!< if `true`, the gradient with respect to the input tensor is computed as well
    const bool realValuedInput;                                 //!< if `true`, the input tensor is real-valued (contains the real part only)
    std::mutex access;

   public:
    /**
     * @brief Instantiates convolution operator.
     * @param device            A device instance
     * @param descriptor        Operation descriptor
     */
    UpstrideConv2DGradFunctor(Device& device, const Conv2DBwdDescriptor& descriptor):
        device(device),
        algebra(descriptor.getAlgebra()),
        convOp(device.getContext(), descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation(), descriptor.isInputGradientRequired()),
        quatKernelOp(device.getContext(), algebra, descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation()),
        requireInputGrad(descriptor.isInputGradientRequired()),
        realValuedInput(descriptor.isRealValuedInput())
    {}

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
        // Sometimes TF sends us an empty tensor, cudnn is not allowed to managed this case so let's avoid it.
        if (inputTensor.getShape().empty()) {
            return;
        }

        // lock access to convOp
        std::lock_guard<std::mutex> lock(access);

        convOp.configure(
            device,
            realValuedInput ? inputTensor.getShape() : inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            kernelTensor.getShape().slice(-4),
            gradTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
            padBefore,
            padAfter,
            groups);

        if (algebra == Algebra::QUATERNION && !realValuedInput) {
            quatKernelOp.configure(
                device,
                inputTensor.getShape(),
                kernelTensor.getShape(),
                padBefore,
                padAfter,
                groups);
        }

        proceedWithAlgebra(algebra, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>& gradTensor,
                            Tensor<Device, T>& kernelGradTensor,
                            Tensor<Device, T>& inputGradTensor) {
        MemoryRequest memory(device, *this);

        // run custom convolution kernels for quaternions if possible
        if (quatKernelOp.canRun()) {
            quatKernelOp(device, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
        }

        else if (realValuedInput && algebra != Algebra::REAL) {
            using CliffordProductSpec = CliffordProductSpec<algebra>;

            // split tensors along blades
            const TensorSplit<Device, const T, CliffordProductSpec::DIMS> kernel(kernelTensor, false), grad(gradTensor);
            const auto& input(inputTensor);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> kernelGrad(kernelGradTensor);

            // check for input gradient requiredness
            if (requireInputGrad) {
                // Computing gradient of a hypercomplex quantity with respect to a real variable; no reason for it to be real
                // However, the imaginary parts are not even stored. The input gradient cannot be required then.
                throw std::logic_error("Input gradient required for type0 input tensor");
            }

            // allocate a temporary buffer
            TemporaryTensor<Device, T> bufferKernel(device, memory, kernelGrad.shape());

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();
            bufferKernel.prepare();

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::productBackprop(
                [this, &memory, &input, &kernel, &grad, &kernelGrad, &inputGradTensor](int left, int right, int dim) {
                    if (left == 0)
                        convOp(input, kernel[right], grad[dim], kernelGrad[right], inputGradTensor);
                    else {
                        // input is real: if left != 0, it is zero
                        kernelGrad[right].zero();
                    }
                },

                [this, &memory, &input, &kernel, &grad, &kernelGrad, &inputGradTensor, &bufferKernel](int left, int right, int dim, bool positive) {
                    if (left == 0) {
                        convOp(input, kernel[right], grad[dim], bufferKernel, inputGradTensor);
                        if (positive)
                            kernelGrad[right] += bufferKernel;
                        else
                            kernelGrad[right] -= bufferKernel;
                    }
                    // nothing to do in else branch since the input is real: if left != 0, it is zero
                });
        }

        // factorized quaternion convolution fallback
        else if (algebra == Algebra::QUATERNION && device.getContext().preferSpeedToMemory()) {
            // split tensors along blades
            const TensorSplit<Device, const T, 4>
                input(inputTensor),
                kernel(kernelTensor, false),
                grad(gradTensor);
            TensorSplit<Device, T, 4> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

            // get temporary buffers
            std::array<TemporaryTensor<Device, T>, 8> inputLanes{ {
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() },
                { device, memory, input.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> kernelLanes{ {
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() },
                { device, memory, kernel.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> gradLanes{ {
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() },
                { device, memory, grad.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> kernelGradLanes{ {
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() },
                { device, memory, kernelGrad.shape() }
            } };

            std::array<TemporaryTensor<Device, T>, 8> inputGradLanes{ {
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() },
                { device, memory, inputGrad.shape() }
            } };

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();

            for (int i = 0; i < 8; ++i) {
                inputLanes[i].prepare();
                kernelLanes[i].prepare();
                gradLanes[i].prepare();
                kernelGradLanes[i].prepare();
                inputGradLanes[i].prepare();
            }

            // decompose - compute - recompose
            TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes.data(), kernel, kernelLanes.data());
            TensorManipulations<Device>::decomposeQuaternionOutputGrad(grad, gradLanes.data());

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();

            for (int i = 0; i < 8; ++i)
                convOp(inputLanes[i], kernelLanes[i], gradLanes[i], kernelGradLanes[i], inputGradLanes[i]);

            TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes.data(), inputGrad, kernelGradLanes.data(), kernelGrad);
        }

        // generic implementation
        else {
            using CliffordProductSpec = CliffordProductSpec<algebra>;

            // split tensors along blades
            const TensorSplit<Device, const T, CliffordProductSpec::DIMS>
                input(inputTensor),
                kernel(kernelTensor, kernelTensor.getShape().getSize() == 4),
                grad(gradTensor);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

            // allocate a temporary buffer
            TemporaryTensor<Device, T> bufferKernel(device, memory, kernelGrad.shape());
            TemporaryTensor<Device, T> bufferInput(device, memory, inputGrad.shape());

            // prepare the scalar operation
            convOp.prepare(memory);

            // submit memory request
            memory.submit();
            bufferKernel.prepare();
            bufferInput.prepare();

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::productBackprop(
                [this, &memory, &input, &kernel, &grad, &kernelGrad, &inputGrad](int left, int right, int dim) {
                    convOp(input[left], kernel[right], grad[dim], kernelGrad[right], inputGrad[left]);
                },

                [this, &memory, &input, &kernel, &grad, &kernelGrad, &inputGrad, &bufferKernel, &bufferInput](int left, int right, int dim, bool positive) {
                    convOp(input[left], kernel[right], grad[dim], bufferKernel, bufferInput);
                    if (positive) {
                        kernelGrad[right] += bufferKernel;
                        if (requireInputGrad)
                            inputGrad[left] += bufferInput;
                    } else {
                        kernelGrad[right] -= bufferKernel;
                        if (requireInputGrad)
                            inputGrad[left] -= bufferInput;
                    }
                });
        }
    }
};

}  // namespace upstride