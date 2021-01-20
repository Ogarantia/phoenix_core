/**
 * @file dense.hpp
 * @author Philipe Moura (philipe.moura@upstride.io)
 * @brief Dense layer implementations for different algebras
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <mutex>
#include "algebra_select_mixin.hpp"
#include "algebras.hpp"
#include "backend/api.h"
#include "deferred_allocator.hpp"
#include "utils.hpp"

namespace upstride {

    template <typename Device, typename T>
    class UpstrideDenseFunctor : public AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>> {
        using AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>>::proceedWithAlgebra;

    private:
        Context& context;
        const Algebra algebra;
        std::mutex access;
        ScalarDenseFunctor<Device, T> denseOp;                                      //!< scalar dense operator to be used to implement other data types
        DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], outputLanes[8]; //!< deferred allocators for the factorized quaternion implementation
        DeferredAllocator<Device, T> buffer;                                        //!< deferred allocator for an intermediate buffer for the default implementation

    public:
        /**
         * @brief Instantiate Dense layer operation
         * @param               A context instance
         * @param algebra       Algebra used to compute the dense. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
         * @param dataFormat    Expected tensors format
         */
        UpstrideDenseFunctor(Context& context, Algebra algebra, DataFormat dataFormat, bool useBias):
            context(context),
            algebra(algebra),
            denseOp(context, dataFormat, useBias)
        {}

        /**
         * @brief Executes the dense operation
         * This function may be called from multiple threads.
         * @param inputTensor       Input tensor
         * @param kernelTensor      Kernel tensor
         * @param biasTensor        Pointer to bias tensor; may be null
         * @param outputTensor      Output tensor
         */
        void operator()(Device& device,
                        const Tensor<Device, const T> &inputTensor,
                        const Tensor<Device, const T> &kernelTensor,
                        const Tensor<Device, const T> *biasTensor,
                        Tensor<Device, T> &outputTensor) {
            // Sometimes TF sends us an empty tensor, cudnn is not allowed to managed this case so let's avoid it. 
            if (inputTensor.getShape().empty())
                return;

            // lock access to denseOp
            std::lock_guard<std::mutex> lock(access);

            denseOp.configure(device,
                              inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
                              kernelTensor.getShape().slice(-2),
                              biasTensor ? biasTensor->getShape().split(MULTIVECTOR_DIM[algebra]) : Shape(),
                              outputTensor.getShape().split(MULTIVECTOR_DIM[algebra]));

            proceedWithAlgebra(algebra, device, inputTensor, kernelTensor, biasTensor, outputTensor);
        }

        template <Algebra algebra>
        void proceedWithAlgebra(Device& device,
                                const Tensor<Device, const T> &inputTensor,
                                const Tensor<Device, const T> &kernelTensor,
                                const Tensor<Device, const T> *biasTensor,
                                Tensor<Device, T> &outputTensor) {
            // factorized quaternion fallback
            if (algebra == Algebra::QUATERNION && context.preferSpeedToMemory()) {
                // split tensors along blades
                const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
                TensorSplit<Device, T, 4> output(outputTensor);

                // allocate bias tensor split if the bias tensor is provided
                TensorSplit<Device, const T, 4> *bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

                // get temporary buffers
                AllocatedTensor<Device, T> *inputLanes[8], *kernelLanes[8], *outputLanes[8];
                for (int i = 0; i < 8; ++i)
                {
                    inputLanes[i] = &this->inputLanes[i].get(device, input.shape());
                    kernelLanes[i] = &this->kernelLanes[i].get(device, kernel.shape());
                    outputLanes[i] = &this->outputLanes[i].get(device, output.shape());
                }

                // decompose - compute - recompose
                TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
                for (int i = 0; i < 4; ++i)
                    denseOp(*inputLanes[i], *kernelLanes[i], nullptr, *outputLanes[i]);
                for (int i = 4; i < 8; ++i)
                    // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                    // Adding bias there then!
                    denseOp(*inputLanes[i], *kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, *outputLanes[i]);

                TensorManipulations<Device>::recomposeQuaternionOutput(outputLanes, output);
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
                AllocatedTensor<Device, T> &buffer(this->buffer.get(device, output.shape()));

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
    class UpstrideDenseGradFunctor : public AlgebraSelectionMixin<UpstrideDenseGradFunctor<Device, T>> {
        using AlgebraSelectionMixin<UpstrideDenseGradFunctor<Device, T>>::proceedWithAlgebra;

    private:
        Context& context;
        const Algebra algebra;
        std::mutex access;
        ScalarDenseGradFunctor<Device, T> denseOp;  //!< scalar convolution operator to be used to implement other data types
        DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], gradLanes[8], kernelGradLanes[8], inputGradLanes[8];  //!< deferred allocators for the factorized quaternion implementation
        DeferredAllocator<Device, T> bufferInput, bufferKernel;                                                           //!< deferred allocator for an intermediate buffer for the default implementation
        bool requireInputGrad;                      //!< if `true`, the input gradient is computed

    public:
        /**
         * @brief Instantiates Dense layer gradient operator
         * @param context       A context instance
         * @param algebra       Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
         * @param dataFormat    Expected tensors format
         * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed as well
         */
        UpstrideDenseGradFunctor(Context& context, Algebra algebra, DataFormat dataFormat, bool requireInputGrad):
            context(context),
            algebra(algebra),
            denseOp(context, dataFormat, requireInputGrad),
            requireInputGrad(requireInputGrad)
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
        void operator()(Device& device,
                        const Tensor<Device, const T>& inputTensor,
                        const Tensor<Device, const T>& kernelTensor,
                        const Tensor<Device, const T>& gradTensor,
                        Tensor<Device, T>& kernelGradTensor,
                        Tensor<Device, T>& inputGradTensor) {
            // Sometimes TF sends us an empty tensor, cudnn is not allowed to managed this case so let's avoid it. 
            if (inputTensor.getShape().empty())
                return;

            // lock access to denseOp
            std::lock_guard<std::mutex> lock(access);

            denseOp.configure(device,
                              inputTensor.getShape().split(MULTIVECTOR_DIM[algebra]),
                              kernelTensor.getShape().slice(-2),
                              gradTensor.getShape().split(MULTIVECTOR_DIM[algebra]));

            proceedWithAlgebra(algebra, device, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
        }

        template <Algebra algebra>
        void proceedWithAlgebra(Device& device,
                                const Tensor<Device, const T>& inputTensor,
                                const Tensor<Device, const T>& kernelTensor,
                                const Tensor<Device, const T>& gradTensor,
                                Tensor<Device, T>& kernelGradTensor,
                                Tensor<Device, T>& inputGradTensor) {
            // factorized quaternion fallback
            if (algebra == Algebra::QUATERNION && context.preferSpeedToMemory()) {
                // split tensors along blades
                const TensorSplit<Device, const T, 4>
                    input(inputTensor),
                    kernel(kernelTensor, false),
                    grad(gradTensor);
                TensorSplit<Device, T, 4> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

                // get temporary buffers
                AllocatedTensor<Device, T>*inputLanes[8], *kernelLanes[8], *gradLanes[8], *kernelGradLanes[8], *inputGradLanes[8];
                for (int i = 0; i < 8; ++i) {
                    inputLanes[i] = &this->inputLanes[i].get(device, input.shape());
                    kernelLanes[i] = &this->kernelLanes[i].get(device, kernel.shape());
                    gradLanes[i] = &this->gradLanes[i].get(device, grad.shape());
                    kernelGradLanes[i] = &this->kernelGradLanes[i].get(device, kernelGrad.shape());
                    inputGradLanes[i] = &this->inputGradLanes[i].get(device, inputGrad.shape());
                }

                // decompose - compute - recompose
                TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
                TensorManipulations<Device>::decomposeQuaternionOutputGrad(grad, gradLanes);

                for (int i = 0; i < 8; ++i)
                    denseOp(*inputLanes[i], *kernelLanes[i], *gradLanes[i], *kernelGradLanes[i], *inputGradLanes[i]);

                TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes, inputGrad, kernelGradLanes, kernelGrad);
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

                // allocate a temporary buffer
                AllocatedTensor<Device, T>& bufferKernel(this->bufferKernel.get(device, kernelGrad.shape()));
                AllocatedTensor<Device, T>& bufferInput(this->bufferInput.get(device, inputGrad.shape()));
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
