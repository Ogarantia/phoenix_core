/**
 * @file dense.hpp
 * @author Philipe Moura (philipe.moura@upstride.io)
 * @brief Dense layer implementations for different algebras
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <mutex>
#include "backend/dense_descriptor.hpp"
#include "algebra_select_mixin.hpp"
#include "algebras.hpp"
#include "backend/api.h"
#include "backend/operation.hpp"
#include "backend/temporary_tensor.hpp"
#include "utils.hpp"

namespace upstride {

    template <typename Device, typename T>
    class UpstrideDenseFunctor : public AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>>, public Operation {
        using AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>>::proceedWithAlgebra;

    private:
        Device& device;                                                             //!< the device instance the operation is attached to
        const Algebra algebra;
        ScalarDenseFunctor<Device, T> denseOp;                                      //!< scalar dense operator to be used to implement other data types

    public:
        UpstrideDenseFunctor(Device& device, const DenseFwdDescriptor& descriptor):
            device(device),
            algebra(descriptor.getAlgebra()),
            denseOp(device.getContext(), descriptor.getDataFormat(), descriptor.isBiasUsed())
        {}

        /**
         * @brief Executes the dense operation
         * This function may be called from multiple threads.
         * @param inputTensor       Input tensor
         * @param kernelTensor      Kernel tensor
         * @param biasTensor        Pointer to bias tensor; may be null
         * @param outputTensor      Output tensor
         */
        void operator()(const Tensor<Device, const T> &inputTensor,
                        const Tensor<Device, const T> &kernelTensor,
                        const Tensor<Device, const T> *biasTensor,
                        Tensor<Device, T> &outputTensor) {
            // Sometimes TF sends us an empty tensor, cudnn is not allowed to managed this case so let's avoid it.
            if (inputTensor.getShape().empty())
                return;

            denseOp.configure(device,
                              inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
                              kernelTensor.getShape().slice(-2),
                              biasTensor ? biasTensor->getShape().split(MULTIVECTOR_DIM[algebra]) : Shape(),
                              outputTensor.getShape().split(MULTIVECTOR_DIM[algebra]));

            proceedWithAlgebra(algebra, inputTensor, kernelTensor, biasTensor, outputTensor);
        }

        template <Algebra algebra>
        void proceedWithAlgebra(const Tensor<Device, const T> &inputTensor,
                                const Tensor<Device, const T> &kernelTensor,
                                const Tensor<Device, const T> *biasTensor,
                                Tensor<Device, T> &outputTensor) {
            MemoryRequest memory(device, *this);

            if (algebra == REAL) {
                denseOp(inputTensor, kernelTensor, biasTensor, outputTensor);
            }

            // factorized quaternion fallback
            else if (algebra == Algebra::QUATERNION && device.getContext().preferSpeedToMemory()) {
                // split tensors along blades
                const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
                TensorSplit<Device, T, 4> output(outputTensor);

                // allocate bias tensor split if the bias tensor is provided
                TensorSplit<Device, const T, 4> *bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

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
                    denseOp(inputLanes[i], kernelLanes[i], nullptr, outputLanes[i]);
                for (int i = 4; i < 8; ++i)
                    // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                    // Adding bias there then!
                    denseOp(inputLanes[i], kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, outputLanes[i]);

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
                    kernel(kernelTensor, kernelTensor.getShape().getSize() == 2);
                TensorSplit<Device, T, CliffordProductSpec::DIMS> output(outputTensor);

                // allocate bias tensor split if the bias tensor is provided
                TensorSplit<Device, const T, CliffordProductSpec::DIMS> *bias = biasTensor ? new TensorSplit<Device, const T, CliffordProductSpec::DIMS>(*biasTensor) : nullptr;

                // allocate a temporary buffer
                TemporaryTensor<Device, T> buffer(device, memory, output.shape());

                // submit memory request
                memory.submit();

                // prepare buffer
                buffer.prepare();

                // compute the Clifford product
                BinaryOperation<CliffordProductSpec>::product(
                    [this, &input, &kernel, bias, &output](int left, int right, int dim) {
                        denseOp(input[left], kernel[right], bias ? &(*bias)[dim] : nullptr, output[dim]);
                    },
                    [this, &input, &kernel, &output, &buffer](int left, int right, int dim, bool positive) {
                        denseOp(input[left], kernel[right], nullptr, buffer);
                        if (positive)
                            output[dim] += buffer;
                        else
                            output[dim] -= buffer;
                    }
                );

                // free bias
                delete bias;
            }
        }
    };

    template <typename Device, typename T>
    class UpstrideDenseGradFunctor : public AlgebraSelectionMixin<UpstrideDenseGradFunctor<Device, T>>, public Operation {
        using AlgebraSelectionMixin<UpstrideDenseGradFunctor<Device, T>>::proceedWithAlgebra;

    private:
        Device& device;                                                                                                   //!< the device instance the operation is attached to
        const Algebra algebra;
        ScalarDenseGradFunctor<Device, T> denseOp;                                                                        //!< scalar convolution operator to be used to implement other data types
        bool requireInputGrad;                      //!< if `true`, the input gradient is computed

    public:
        /**
         * @brief Instantiates Dense layer gradient operator
         * @param device        A device instance
         * @param algebra       Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
         * @param dataFormat    Expected tensors format
         * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed as well
         */
        UpstrideDenseGradFunctor(Device& device, const DenseBwdDescriptor& descriptor):
            device(device),
            algebra(descriptor.getAlgebra()),
            denseOp(device.getContext(), descriptor.getDataFormat(), descriptor.isInputGradientRequired()),
            requireInputGrad(descriptor.isInputGradientRequired())
        {}

        /**
         * @brief Executes the operation
         * This function may be called from multiple threads.
         * @param inputTensor       forward input tensor
         * @param kernelTensor      forward input kernel tensor
         * @param gradTensor        gradient of the forward output tensor (dy)
         * @param kernelGradTensor  output: kernel gradient
         * @param inputGradTensor   output: input gradient
         */
        void operator()(const Tensor<Device, const T>& inputTensor,
                        const Tensor<Device, const T>& kernelTensor,
                        const Tensor<Device, const T>& gradTensor,
                        Tensor<Device, T>& kernelGradTensor,
                        Tensor<Device, T>& inputGradTensor) {
            // Sometimes TF sends us an empty tensor, cudnn is not allowed to managed this case so let's avoid it.
            if (inputTensor.getShape().empty())
                return;

            denseOp.configure(device,
                              inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
                              kernelTensor.getShape().slice(-2),
                              gradTensor.getShape().split(MULTIVECTOR_DIM[algebra]));

            proceedWithAlgebra(algebra, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
        }

        template <Algebra algebra>
        void proceedWithAlgebra(const Tensor<Device, const T>& inputTensor,
                                const Tensor<Device, const T>& kernelTensor,
                                const Tensor<Device, const T>& gradTensor,
                                Tensor<Device, T>& kernelGradTensor,
                                Tensor<Device, T>& inputGradTensor) {
            MemoryRequest memory(device, *this);

            if (algebra == REAL) {
                denseOp(inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
            }

            // factorized quaternion fallback
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

                for (int i = 0; i < 8; ++i)
                    denseOp(inputLanes[i], kernelLanes[i], gradLanes[i], kernelGradLanes[i], inputGradLanes[i]);

                TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes.data(), inputGrad, kernelGradLanes.data(), kernelGrad);
            }

            // generic implementation
            else {
                using CliffordProductSpec = CliffordProductSpec<algebra>;

                // split tensors along blades
                const TensorSplit<Device, const T, CliffordProductSpec::DIMS>
                    input(inputTensor),
                    kernel(kernelTensor, kernelTensor.getShape().getSize() == 2),
                    grad(gradTensor);
                TensorSplit<Device, T, CliffordProductSpec::DIMS> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

                // allocate a temporary buffers
                TemporaryTensor<Device, T> bufferKernel(device, memory, kernelGrad.shape());
                TemporaryTensor<Device, T> bufferInput(device, memory, inputGrad.shape());

                // submit the memory request
                memory.submit();

                // prepare buffers
                bufferKernel.prepare();
                bufferInput.prepare();

                // compute the Clifford product
                BinaryOperation<CliffordProductSpec>::productBackprop(
                    [this, &input, &kernel, &grad, &kernelGrad, &inputGrad](int left, int right, int dim) {
                        denseOp(input[left], kernel[right], grad[dim], kernelGrad[right], inputGrad[left]);
                    },
                    [this, &input, &kernel, &grad, &kernelGrad, &inputGrad, &bufferKernel, &bufferInput](int left, int right, int dim, bool positive) {
                        denseOp(input[left], kernel[right], grad[dim], bufferKernel, bufferInput);
                        if (positive) {
                            kernelGrad[right] += bufferKernel;
                            if (requireInputGrad)
                                inputGrad[left] += bufferInput;
                        } else {
                            kernelGrad[right] -= bufferKernel;
                            if (requireInputGrad)
                                inputGrad[left] -= bufferInput;
                        }
                    }
                );
            }
        }
    };
} // namespace upstride
