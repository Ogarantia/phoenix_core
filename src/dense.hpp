/**
 * @file dense.hpp
 * @author Philipe Moura (philipe.moura@upstride.io)
 * @brief Dense layer implementations for different algebras
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

    template <typename Device, typename T>
    class UpstrideDenseFunctor : public AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>> {
        using AlgebraSelectionMixin<UpstrideDenseFunctor<Device, T>>::proceedWithAlgebra;

    private:
        Context& context;
        ThreadLocalPtr<ScalarDenseFunctor<Device, T>> denseOp; //!< scalar dense operator to be used to implement other data types
        Algebra algebra;
        DataFormat dataFormat;
        DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], outputLanes[8]; //!< deferred allocators for the factorized quaternion implementation
        DeferredAllocator<Device, T> buffer;                                        //!< deferred allocator for an intermediate buffer for the default implementation

    public:
        UpstrideDenseFunctor(Context& context): context(context) { }

        /**
         * @brief Sets main dense parameters indepentent from the input, filter and output sizes
         * @param algebra       Algebra used to compute the dense. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
         * @param dataFormat    Expected tensors format
         */
        void configure(Algebra algebra, DataFormat dataFormat) {
            this->algebra = algebra;
            this->dataFormat = dataFormat;
        }

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
            // ensure the object exists within the current thread
            denseOp(context, dataFormat, biasTensor != nullptr);
            denseOp->configure(
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
            if (algebra == Algebra::QUATERNION) {
                // split tensors along blades
                const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
                TensorSplit<Device, T, 4> output(outputTensor);

                // allocate bias tensor split if the bias tensor is provided
                TensorSplit<Device, const T, 4> *bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

                // get temporary buffers
                AllocatedTensor<Device, T> *inputLanes[8], *kernelLanes[8], *outputLanes[8];
                for (int i = 0; i < 8; ++i)
                {
                    inputLanes[i] = &this->inputLanes[i].get(inputTensor.getDevice(), input.shape());
                    kernelLanes[i] = &this->kernelLanes[i].get(kernelTensor.getDevice(), kernel.shape());
                    outputLanes[i] = &this->outputLanes[i].get(outputTensor.getDevice(), output.shape());
                }

                // decompose - compute - recompose
                TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
                for (int i = 0; i < 4; ++i)
                    (*denseOp)(*inputLanes[i], *kernelLanes[i], nullptr, *outputLanes[i]);
                for (int i = 4; i < 8; ++i)
                    // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                    // Adding bias there then!
                    (*denseOp)(*inputLanes[i], *kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, *outputLanes[i]);

                TensorManipulations<Device>::recomposeQuaternionOutput(outputLanes, output);
                // free bias
                delete bias;
            }

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
                AllocatedTensor<Device, T> &buffer(this->buffer.get(outputTensor.getDevice(), output.shape()));

                // compute the Clifford product
                BinaryOperation<CliffordProductSpec>::product(
                    [this, &input, &kernel, bias, &output](int left, int right, int dim) {
                        (*denseOp)(input[left], kernel[right], bias ? &(*bias)[dim] : nullptr, output[dim]);
                    },
                    [this, &input, &kernel, &output, &buffer](int left, int right, int dim, bool positive) {
                        (*denseOp)(input[left], kernel[right], nullptr, buffer);
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
        ThreadLocalPtr<ScalarDenseGradFunctor<Device, T>> denseOp;  //!< scalar convolution operator to be used to implement other data types
        Algebra algebra;
        DataFormat dataFormat;
        bool requireInputGrad;
        DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], gradLanes[8], kernelGradLanes[8], inputGradLanes[8];  //!< deferred allocators for the factorized quaternion implementation
        DeferredAllocator<Device, T> bufferInput, bufferKernel;                                                           //!< deferred allocator for an intermediate buffer for the default implementation

    public:
        UpstrideDenseGradFunctor(Context& context): context(context) { }

        /**
         * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
         * @param algebra       Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
         * @param dataFormat    Expected tensors format
         * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed as well
         */
        void configure(Algebra algebra, DataFormat dataFormat, bool requireInputGrad) {
            this->algebra = algebra;
            this->dataFormat = dataFormat;
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
         */
        void operator()(const Tensor<Device, const T>& inputTensor,
                        const Tensor<Device, const T>& kernelTensor,
                        const Tensor<Device, const T>& gradTensor,
                        Tensor<Device, T>& kernelGradTensor,
                        Tensor<Device, T>& inputGradTensor) {
            // ensure the object exists within the current thread
            denseOp(context, dataFormat, requireInputGrad);
            denseOp->configure(
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
            if (algebra == Algebra::QUATERNION) {
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
                    (*denseOp)(*inputLanes[i], *kernelLanes[i], *gradLanes[i], *kernelGradLanes[i], *inputGradLanes[i]);

                TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes, inputGrad, kernelGradLanes, kernelGrad);
            }

            else {
                using CliffordProductSpec = CliffordProductSpec<algebra>;

                // split tensors along blades
                const TensorSplit<Device, const T, CliffordProductSpec::DIMS>
                    input(inputTensor),
                    kernel(kernelTensor, kernelTensor.getShape().getSize() == 2),
                    grad(gradTensor);
                TensorSplit<Device, T, CliffordProductSpec::DIMS> kernelGrad(kernelGradTensor), inputGrad(inputGradTensor);

                // allocate a temporary buffer
                AllocatedTensor<Device, T>& bufferKernel(this->bufferKernel.get(kernelGradTensor.getDevice(), kernelGrad.shape()));
                AllocatedTensor<Device, T>& bufferInput(this->bufferInput.get(inputGradTensor.getDevice(), inputGrad.shape()));
                // compute the Clifford product
                BinaryOperation<CliffordProductSpec>::productBackprop(
                    [this, &input, &kernel, &grad, &kernelGrad, &inputGrad](int left, int right, int dim) {
                        (*denseOp)(input[left], kernel[right], grad[dim], kernelGrad[dim], inputGrad[dim]);
                    },
                    [this, &input, &kernel, &grad, &kernelGrad, &inputGrad, &bufferKernel, &bufferInput](int left, int right, int dim, bool positive) {
                        (*denseOp)(input[left], kernel[right], grad[dim], bufferKernel, bufferInput);
                        if (positive) {
                            kernelGrad[dim] += bufferKernel;
                            inputGrad[dim] += bufferInput;
                        } else {
                            kernelGrad[dim] -= bufferKernel;
                            inputGrad[dim] -= bufferInput;
                        }
                    }
                );
            }
        }
    };
} // namespace upstride