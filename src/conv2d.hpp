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
#include "deferred_allocator.hpp"
#include "utils.hpp"

namespace upstride {

/**
 * @brief Specifies memory layout for a convolution kernel
 *   OIHW for real algebra
 *   nOIHW for a non-real algebra containing n-dimensional multivectors.
 */
class Conv2DKernelLayout {
   public:
    /**
     * @brief Returns number of dimensions in the kernel tensor for a specific algebra
     */
    static inline int rank(Algebra algebra) {
        return algebra == Algebra::REAL ? 4 : 5;
    }

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


/**
 * @brief Describes 2D convolution operation parameters
 */
class Conv2DDescriptor {
private:
    const Shape inputShape;             //!< operation input tensor shape
    const Shape filterShape;            //!< operation filter tensor shape
    const IntPair stride;               //!< strides along H and W dimensions in pixels
    const IntPair dilation;             //!< dilations along H and W dimensions in pixels
    const int groups;                   //!< number of groups (for group convolutions)
    const Algebra algebra;              //!< algebra corresponding to UpStride datatype
    const DataFormat dataFormat;        //!< input and output tensors data format (channels-first or channels-last)
    const bool useBias;                 //!< if `true`, the bias addition is enabled
    const bool realValuedInput;         //!< if `true`, the input tensor is real-valued
    IntPair padBefore;                  //!< top-left zero-padding applied to the input along H and W dimensions in pixels
    IntPair padAfter;                   //!< bottom-right zero-padding applied to the input along H and W dimensions in pixels

public:
    Conv2DDescriptor(
        const Shape& inputShape,
        const Shape& filterShape,
        IntPair stride,
        IntPair dilation,
        Padding paddingPreset,
        const IntTuple& explicitPadding,
        int groups,
        Algebra algebra,
        DataFormat dataFormat,
        bool useBias,
        bool realValuedInput = false
    ):
        inputShape(inputShape), filterShape(filterShape),
        stride(stride), dilation(dilation), groups(groups), algebra(algebra), dataFormat(dataFormat), useBias(useBias), realValuedInput(realValuedInput)
    {
        // Perform shape checks
        if (inputShape.getSize() != 4)
            throw std::invalid_argument("Four-dimensional input tensor expected");
        if (algebra != Algebra::REAL) {
            if (filterShape.getSize() != 5)
                throw std::invalid_argument("Five-dimensional filter tensor expected");
            if (filterShape[0] != MULTIVECTOR_DIM[algebra])
                throw std::invalid_argument("First filter dimension mismatch, got " + std::to_string(filterShape[0]));
        } else if (filterShape.getSize() != 4)
            throw std::invalid_argument("Four-dimensional filter tensor expected");

        // init padding
        if (paddingPreset == Padding::EXPLICIT) {
            padBefore = IntPair(explicitPadding[0], explicitPadding[1]);
            padAfter = IntPair(explicitPadding[2], explicitPadding[3]);
        }

        // compute padding
        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filterShape[Conv2DKernelLayout::heightDim(algebra)],
            dilation.x, stride.x, paddingPreset,
            padBefore.x, padAfter.x);

        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filterShape[Conv2DKernelLayout::widthDim(algebra)],
            dilation.y, stride.y, paddingPreset,
            padBefore.y, padAfter.y);
    }

    /**
     * @brief Computes the output tensor shape
     */
    inline Shape getOutputShape() const {
        // Set up the resulting shape
        Shape outputShape(4);
        outputShape[0] = inputShape[0];
        outputShape.depth(dataFormat) = filterShape[Conv2DKernelLayout::numOutputChannelsDim(algebra)];

        // compute output size
        outputShape.height(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filterShape[Conv2DKernelLayout::heightDim(algebra)],
            dilation.x, stride.x,
            padBefore.x, padAfter.x);

        outputShape.width(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filterShape[Conv2DKernelLayout::widthDim(algebra)],
            dilation.y, stride.y,
            padBefore.y, padAfter.y);

        // in case of real-valued input, the output batch size is N times bigger
        if (realValuedInput)
            outputShape[0] = outputShape[0] * upstride::MULTIVECTOR_DIM[algebra];

        return outputShape;
    }

    inline bool operator==(const Conv2DDescriptor& another) const {
        return inputShape == another.inputShape &&
               filterShape == another.filterShape &&
               stride == another.stride &&
               dilation == another.dilation &&
               groups == another.groups &&
               algebra == another.algebra &&
               dataFormat == another.dataFormat &&
               useBias == another.useBias &&
               realValuedInput == another.realValuedInput &&
               padBefore == another.padBefore &&
               padAfter == another.padAfter;
    }


    template<typename T>
    static inline bool compareHelper(const T& a, const T& b, bool& result) {
        if (a < b) {
            result = true;
            return true;
        }
        if (b < a) {
            return true;
        }
        return false;
    }


    inline bool operator<(const Conv2DDescriptor& another) const {
        if (inputShape < another.inputShape) return true;
        if (another.inputShape < inputShape) return false;

        if (filterShape < another.filterShape) return true;
        if (another.filterShape < filterShape) return false;

        if (stride < another.stride) return true;
        if (another.stride < stride) return false;

        if (dilation < another.dilation) return true;
        if (another.dilation < dilation) return false;

        if (groups < another.groups) return true;
        if (another.groups < groups) return false;

        if (algebra < another.algebra) return true;
        if (another.algebra < algebra) return false;

        if (dataFormat < another.dataFormat) return true;
        if (another.dataFormat < dataFormat) return false;

        if (useBias < another.useBias) return true;
        if (another.useBias < useBias) return false;

        if (realValuedInput < another.realValuedInput) return true;
        if (another.realValuedInput < realValuedInput) return false;

        if (padBefore < another.padBefore) return true;
        if (another.padBefore < padBefore) return false;

        if (padAfter < another.padAfter) return true;
        if (another.padAfter < padAfter) return false;
    }


    inline const IntPair& getStride() const { return stride; }

    inline const IntPair& getDilation() const { return dilation; }

    inline int getGroups() const { return groups; }

    inline Algebra getAlgebra() const { return algebra; }

    inline DataFormat getDataFormat() const { return dataFormat; }

    inline bool isRealValuedInput() const { return realValuedInput; }

    inline bool isBiasUsed() const { return useBias; }

    inline IntPair getPaddingBefore() const { return padBefore; }

    inline IntPair getPaddingAfter() const { return padAfter; }
};

template <typename Device, typename T>
class UpstrideConv2DFunctor : public AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>> {
    using AlgebraSelectionMixin<UpstrideConv2DFunctor<Device, T>>::proceedWithAlgebra;

   private:
    Context& context;                           //!< a global context the operation belongs to
    const Algebra algebra;
    ScalarConv2DFunctor<Device, T> convOp;      //!< scalar convolution operator to be used to implement other data types
    cuda::QuatKernelPointwiseConvForwardFunctor<Device, T> quatKernelOp;       //!< custom kernels operator for quaternionic pointwise convolution
    DeferredAllocator<Device, T> inputLanes[8], kernelLanes[8], outputLanes[8];  //!< deferred allocators for the factorized quaternion implementation
    DeferredAllocator<Device, T> buffer;                                         //!< deferred allocator for an intermediate buffer for the default implementation
    const bool realValuedInput;                 //!< if `true`, the input tensor is real-valued (contains the real part only)
    std::mutex access;

   public:
    /**
     * @brief Instantiates convolution operator
     * @param context       A context instance
     * @param descriptor    Operation descriptor
     */
    UpstrideConv2DFunctor(Context& context, const Conv2DDescriptor& descriptor):
        context(context),
        algebra(descriptor.getAlgebra()),
        convOp(context, descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation(), descriptor.isBiasUsed()),
        quatKernelOp(context, algebra, descriptor.getDataFormat(), descriptor.getStride(), descriptor.getDilation()),
        realValuedInput(descriptor.isRealValuedInput())
    {}

    /**
     * @brief Executes the convolution operation
     * This function may be called from multiple threads.
     * @param device            A device the operation is computed on
     * @param inputTensor       Input tensor
     * @param kernelTensor      Kernel tensor
     * @param biasTensor        Pointer to bias tensor; may be null
     * @param outputTensor      Output tensor
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void operator()(Device& device,
                    const Tensor<Device, const T>& inputTensor,
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

        proceedWithAlgebra(algebra, device, inputTensor, kernelTensor, biasTensor, outputTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(Device& device,
                            const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>* biasTensor,
                            Tensor<Device, T>& outputTensor) {
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

            // allocate a temporary buffer
            AllocatedTensor<Device, T>& buffer(this->buffer.get(device, output.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::product(
                [this, &input, &kernel, &output](int left, int right, int dim) {
                    if (left == 0)
                        convOp(input, kernel[right], nullptr, output[dim]);
                    // no else branch: input is zero if left > 0, bias is zero as well
                },

                [this, &input, &kernel, &output, &buffer](int left, int right, int dim,  bool positive) {
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
        else if (algebra == Algebra::QUATERNION && context.preferSpeedToMemory()) {
            // split tensors along blades
            const TensorSplit<Device, const T, 4> input(inputTensor), kernel(kernelTensor, false);
            TensorSplit<Device, T, 4> output(outputTensor);

            // allocate bias tensor split if the bias tensor is provided
            TensorSplit<Device, const T, 4>* bias = biasTensor ? new TensorSplit<Device, const T, 4>(*biasTensor) : nullptr;

            // get temporary buffers
            AllocatedTensor<Device, T>*inputLanes[8], *kernelLanes[8], *outputLanes[8];
            for (int i = 0; i < 8; ++i) {
                inputLanes[i] = &this->inputLanes[i].get(device, input.shape());
                kernelLanes[i] = &this->kernelLanes[i].get(device, kernel.shape());
                outputLanes[i] = &this->outputLanes[i].get(device, output.shape());
            }

            // decompose - compute - recompose
            TensorManipulations<Device>::decomposeQuaternionInputs(input, inputLanes, kernel, kernelLanes);
            for (int i = 0; i < 4; ++i)
                convOp(*inputLanes[i], *kernelLanes[i], nullptr, *outputLanes[i]);
            for (int i = 4; i < 8; ++i)
                // According to the factorization formulation, last four lanes are plainly added to the quaternion components (see recomposeQuaternionOutput()).
                // Adding bias there then!
                convOp(*inputLanes[i], *kernelLanes[i], bias ? &(*bias)[i - 4] : nullptr, *outputLanes[i]);

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
                kernel(kernelTensor, kernelTensor.getShape().getSize() == 4);
            TensorSplit<Device, T, CliffordProductSpec::DIMS> output(outputTensor);

            // allocate bias tensor split if the bias tensor is provided
            TensorSplit<Device, const T, CliffordProductSpec::DIMS>* bias = biasTensor ? new TensorSplit<Device, const T, CliffordProductSpec::DIMS>(*biasTensor) : nullptr;

            // allocate a temporary buffer
            AllocatedTensor<Device, T>& buffer(this->buffer.get(device, output.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::product(
                [this, &input, &kernel, bias, &output](int left, int right, int dim) {
                    convOp(input[left], kernel[right], bias ? &(*bias)[dim] : nullptr, output[dim]);
                },

                [this, &input, &kernel, &output, &buffer](int left, int right, int dim,  bool positive) {
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
class UpstrideConv2DGradFunctor : public AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>> {
    using AlgebraSelectionMixin<UpstrideConv2DGradFunctor<Device, T>>::proceedWithAlgebra;

   private:
    Context& context;                                           //!< a global context the operation belongs to
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
     * @brief Instantiates convolution operator
     * @param context               A context instance
     * @param algebra               Algebra used to compute the convolution. The inputs (tensor and filter) are interpreted as matrices of multivectors of this specific algebra.
     * @param dataFormat            Expected tensors format
     * @param stride                Convolution stride
     * @param dilation              Convolution dilation
     * @param requireInputGrad      If `true`, the gradient with respect to the input tensor is computed as well
     * @param realValuedInput       If `true`, the convolution input tensor is real-valued (contains the real part only)
     */
    UpstrideConv2DGradFunctor(Context& context, Algebra algebra, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad, bool realValuedInput = false):
        context(context),
        algebra(algebra),
        convOp(context, dataFormat, stride, dilation, requireInputGrad),
        quatKernelOp(context, algebra, dataFormat, stride, dilation),
        requireInputGrad(requireInputGrad),
        realValuedInput(realValuedInput)
    {}

    /**
     * @brief Executes the operation
     * This function may be called from multiple threads.
     * @param device            A device the operation is computed on
     * @param inputTensor       forward input tensor
     * @param kernelTensor      forward input kernel tensor
     * @param gradTensor        gradient of the forward output tensor (dy)
     * @param kernelGradTensor  output: kernel gradient
     * @param inputGradTensor   output: input gradient
     * @param padBefore         number of zero samples to add to the input tensor on top/left
     * @param padAfter          number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void operator()(Device& device,
                    const Tensor<Device, const T>& inputTensor,
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

        proceedWithAlgebra(algebra, device, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
    }

    template <Algebra algebra>
    void proceedWithAlgebra(Device& device,
                            const Tensor<Device, const T>& inputTensor,
                            const Tensor<Device, const T>& kernelTensor,
                            const Tensor<Device, const T>& gradTensor,
                            Tensor<Device, T>& kernelGradTensor,
                            Tensor<Device, T>& inputGradTensor) {
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
            AllocatedTensor<Device, T>& bufferKernel(this->bufferKernel.get(device, kernelGrad.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::productBackprop(
                [this, &input, &kernel, &grad, &kernelGrad, &inputGradTensor](int left, int right, int dim) {
                    if (left == 0)
                        convOp(input, kernel[right], grad[dim], kernelGrad[right], inputGradTensor);
                    else {
                        // input is real: if left != 0, it is zero
                        kernelGrad[right].zero();
                    }
                },

                [this, &input, &kernel, &grad, &kernelGrad, &inputGradTensor, &bufferKernel](int left, int right, int dim, bool positive) {
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
        else if (algebra == Algebra::QUATERNION && context.preferSpeedToMemory()) {
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
                convOp(*inputLanes[i], *kernelLanes[i], *gradLanes[i], *kernelGradLanes[i], *inputGradLanes[i]);

            TensorManipulations<Device>::recomposeQuaternionInputsGrad(inputGradLanes, inputGrad, kernelGradLanes, kernelGrad);
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
            AllocatedTensor<Device, T>& bufferKernel(this->bufferKernel.get(device, kernelGrad.shape()));
            AllocatedTensor<Device, T>& bufferInput(this->bufferInput.get(device, inputGrad.shape()));

            // compute the Clifford product
            BinaryOperation<CliffordProductSpec>::productBackprop(
                [this, &input, &kernel, &grad, &kernelGrad, &inputGrad](int left, int right, int dim) {
                    convOp(input[left], kernel[right], grad[dim], kernelGrad[right], inputGrad[left]);
                },

                [this, &input, &kernel, &grad, &kernelGrad, &inputGrad, &bufferKernel, &bufferInput](int left, int right, int dim, bool positive) {
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