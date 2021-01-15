/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementation using cuDNN compute backend
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include "../../deferred_allocator.hpp"
#include "../backend.hpp"
#include "context.hpp"
#include "kernels.hpp"

namespace upstride {

/**
 * @brief Selects a conv2d scaling parameter pointer depending on the type of the data the convolution is computed
 * on. cuDNN expects pointers to 32-bit floats for float and half data types, and pointers to doubles otherwise.
 * This function simply selects one of the two values and returns a pointer to it depending on compile time-known
 * tensor data type.
 * https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
 * @tparam T                    The tensor datatype
 * @param singlePrecisionVal    Single precision scaling parameter value
 * @param doublePrecisionVal    Double precision scaling parameter value
 * @return pointer to a scaling parameter value expected by cuDNN.
 */
template <typename T>
static inline const void* selectScalingParameterPtr(const float& singlePrecisionVal, const double& doublePrecisionVal) {
    // default implementation returns a pointer to the single precision value
    return &singlePrecisionVal;
}

template <>
inline const void* selectScalingParameterPtr<double>(const float& singlePrecisionVal, const double& doublePrecisionVal) {
    // specialization for double
    return &doublePrecisionVal;
}

/**
 * @brief Convolution operation base class
 * Keeps common things for forward and backward
 * @tparam T    A scalar datatype of tensors content
 */
class ScalarConv2DBase {
   protected:
    cudnn::Context& context;

    const DataFormat dataFormat;
    const IntPair stride, dilation;

    Shape inputShape, filterShape, outputShape;

    IntPair padBefore;          //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;           //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;          //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;    //!< offset in the output tensor due to the additional padding applied to handle the asymmetric padding
    bool useBuffer;                                      //!< if true, an intermediate buffer is used to store repadded input tensor
    cudnn::Memory scratchpad;                            //!< a memory buffer needed by cuDNN algorithm

    cudnnConvolutionDescriptor_t convDesc;               //!< cuDNN convolution operator descriptor
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;

    ScalarConv2DBase(upstride::Context& context, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation) :
        context(static_cast<cudnn::Context&>(context)), dataFormat(dataFormat), stride(stride), dilation(dilation) {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&outputDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&filterDesc));
    }

    ~ScalarConv2DBase() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
    }

    template <typename datatype>
    void setConvDescriptor(cudnnConvolutionDescriptor_t& descriptor, int groups) {
        cudnn::Context::raiseIfError(cudnnSetConvolution2dDescriptor(
            descriptor, actualPad.x, actualPad.y, stride.x, stride.y, dilation.x, dilation.y, CUDNN_CROSS_CORRELATION,
            cudnn::getDataType<datatype>()));

        if (groups > 1)
            cudnn::Context::raiseIfError(cudnnSetConvolutionGroupCount(descriptor, groups));
    }

    /**
     * @brief For an asymmetric input padding, computes symmetric padding allowing to have the same output tensor as for
     * the original asymmetric padding, up to a crop. Otherwise transmits the input padding as is.
     * @param repaddingOffset   the offset value of the output after the repadding
     * @return the symmetric padding.
     */
    inline IntPair symmetrizePadding(IntPair& repaddingOffset) {
        // Proceed with the symmetric padding covering the requested padding.
        // Adding one step (stride) to padBefore is equivalent to add an entry to output at the beginning of every
        // dimension. This is be cropped further on after the convolution is computed.
        IntPair actualPad;
        if (padBefore.x != padAfter.x) {
            actualPad.x = padBefore.x + stride.x;
            repaddingOffset.x = 1;
        } else {
            actualPad.x = padBefore.x;
            repaddingOffset.x = 0;
        }

        if (padBefore.y != padAfter.y) {
            actualPad.y = padBefore.y + stride.y;
            repaddingOffset.y = 1;
        } else {
            actualPad.y = padBefore.y;
            repaddingOffset.y = 0;
        }

        if (padAfter.x > actualPad.x || padAfter.y > actualPad.y)
            throw std::runtime_error("Cannot handle asymmetric padding");
        return actualPad;
    }
};


/**
 * @brief 2D convolution implementation using cuDNN.
 * @tparam T    A scalar datatype of tensors content
 */
template <typename T>
class ScalarConv2DFunctor<device::CUDA, T> : public ScalarConv2DBase {
   private:
    DeferredAllocator<device::CUDA, T> bufferAllocator;  //!< intermediate buffer to store the repadded input tensor
    Shape repaddedOutputShape;                           //!< shape of the output tensor having an additional symmetrized zero padding
    cudnnConvolutionFwdAlgo_t algorithm;                 //!< cuDNN convolution computation algorithm

   public:
    /**
     * @brief Instantiates a Conv2D operation.
     * Sets main convolution parameters independent from the input, filter and output sizes.
     * @param context       A context instance
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     * @param useBias       If `true`, the bias addition is enabled.
     */
    ScalarConv2DFunctor(upstride::Context& context, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool useBias) :
        ScalarConv2DBase(context, dataFormat, stride, dilation)
    {}

    /**
     * @brief Performs backend-related operation configuration
     * @param device            A device the operation will be executed on
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void configure(device::CUDA& device, const Shape& inputShape, const Shape& filterShape, const Shape& biasShape, const Shape& outputShape, const IntPair& padBefore, const IntPair& padAfter, int groups) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->filterShape == filterShape && this->outputShape == outputShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->filterShape = filterShape;
        this->outputShape = outputShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // check for padding symmetry
        repaddedOutputShape = outputShape;
        if (padBefore == padAfter) {
            actualPad = padBefore;
            useBuffer = false;
        } else {
            actualPad = symmetrizePadding(repaddingOffset);
            repaddedOutputShape.height(dataFormat) += repaddingOffset.x;
            repaddedOutputShape.width(dataFormat) += repaddingOffset.y;
            useBuffer = true;
        }

        // setup tensors
        cudnn::setTensorDescriptor<T>(outputDesc, repaddedOutputShape, dataFormat);
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, dataFormat);

        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,  // OIHW according to the docs
            filterShape[0], filterShape[1], filterShape[2], filterShape[3]));

        // algorithm selection, half float case first
        size_t scratchpadSize;
        float executionTime;
        cudnnMathType_t mathType;
        if (cudnn::isHalfFloat<T>()) {
            // fp32 computing might be faster than fp16 computing and is more accurate, so try fp32 computing first
            setConvDescriptor<float>(convDesc, groups);
            // enable tensor multiplication units (like Tensor Cores)
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
            algorithm = device.selectForwardAlgo(context, convDesc, inputDesc, filterDesc, outputDesc, executionTime, scratchpadSize, mathType);

            if (context.isFp16ConvForwardAllowed()) {
                // fp16 computing is allowed, try it as well
                float fp16ExecTime;
                size_t fp16ScratchpadSize;
                cudnnMathType_t fp16MathType;
                setConvDescriptor<half>(convDesc, groups);
                cudnnConvolutionFwdAlgo_t fp16Algorithm = device.selectForwardAlgo(context, convDesc, inputDesc, filterDesc, outputDesc, fp16ExecTime, fp16ScratchpadSize, fp16MathType);
                if (fp16ExecTime < executionTime) {
                    // fp16 compute is actually faster than fp32, use it
                    UPSTRIDE_SAYS(context, "fp16 is faster than fp32 for forward pass");
                    scratchpadSize = fp16ScratchpadSize;
                    algorithm = fp16Algorithm;
                    mathType = fp16MathType;
                }
                else
                    setConvDescriptor<float>(convDesc, groups);
            }
        }
        // algorithm selection for other datatypes
        else {
            setConvDescriptor<T>(convDesc, groups);
            algorithm = device.selectForwardAlgo(context, convDesc, inputDesc, filterDesc, outputDesc, executionTime, scratchpadSize, mathType);
        }

        // set the math type according to the chosen algorithm
        cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, mathType));

        // allocate scratchpad
        if (scratchpad.getSize() != scratchpadSize)
            scratchpad = cudnn::Memory(scratchpadSize);
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param biasTensor        Pointer to bias tensor; may be null
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<device::CUDA, const T>& inputTensor,
                    const Tensor<device::CUDA, const T>& filterTensor,
                    const Tensor<device::CUDA, const T>* biasTensor,
                    Tensor<device::CUDA, T>& outputTensor) {
        // allocate buffer, if needed
        AllocatedTensor<device::CUDA, T>* buffer = nullptr;
        if (useBuffer)
            buffer = &bufferAllocator.get(outputTensor.getDevice(), repaddedOutputShape);

        // perform the convolution
        const float alphaF = 1.0f, betaF = 0.0f;
        const double alphaD = 1.0, betaD = 0.0;
        cudnn::Context::raiseIfError(cudnnConvolutionForward(
            inputTensor.getDevice().handle(),
            selectScalingParameterPtr<T>(alphaF, alphaD),
            inputDesc, inputTensor.getDataPtr(),
            filterDesc, filterTensor.getDataPtr(),
            convDesc,
            algorithm,
            scratchpad.pointer(), scratchpad.getSize(),
            selectScalingParameterPtr<T>(betaF, betaD),
            outputDesc, useBuffer ? buffer->getDataPtr() : outputTensor.getDataPtr()));

        // crop, if needed
        if (useBuffer)
            cudnn::crop(*buffer, outputTensor, dataFormat, repaddingOffset);

        // add bias
        if (biasTensor)
            cudnn::addBias(outputTensor, *biasTensor, dataFormat);

    }
};  // namespace upstride

/**
 * @brief 2D backward convolution implementation using cuDNN
 * @tparam T    A scalar datatype of tensors content
 */
template <typename T>
class ScalarConv2DGradFunctor<device::CUDA, T> : public ScalarConv2DBase {
   private:
    const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not
    Shape gradShape;
    Shape repaddedGradShape;                             //!< shape of the gradient tensor after additional symmetric zero padding
    DeferredAllocator<device::CUDA, T> bufferAllocator;  //!< ntermediate buffer to store the repadded gradient tensor

    cudnnConvolutionBwdDataAlgo_t inputGradientAlgo;     //!< cuDNN backward algorithm used to compute the input (data) gradient
    cudnnConvolutionBwdFilterAlgo_t kernelGradientAlgo;  //!< cuDNN backward algorithm used to compute the kernel (filter) gradient
    cudnnTensorDescriptor_t gradDesc;                    //!< output value gradient (dy) descriptor (which is an input of this operation)
    cudnnMathType_t inputMathType, kernelMathType;       //!< math types for the chosen algorithms, either regular math or Tensor Cores


   public:
    ScalarConv2DGradFunctor(
        upstride::Context& context, DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) :
        ScalarConv2DBase(context, dataFormat, stride, dilation), requireInputGrad(requireInputGrad) {
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&gradDesc));
    }

    ~ScalarConv2DGradFunctor() {
        cudnnDestroyTensorDescriptor(gradDesc);
    }

    /**
     * @brief Performs backend-related operation configuration
     * @param device            A device the operation will be executed on
     * @param inputShape        Input tensor shape
     * @param filterShape       kernel tensor shape
     * @param gradShape         grad tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void configure(device::CUDA& device,
                   const Shape& inputShape,
                   const Shape& filterShape,
                   const Shape& gradShape,
                   const IntPair& padBefore,
                   const IntPair& padAfter,
                   int groups) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->filterShape == filterShape && this->gradShape == gradShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->filterShape = filterShape;
        this->gradShape = gradShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // check for padding symmetry
        repaddedGradShape = gradShape;
        if (padBefore == padAfter) {
            actualPad = padBefore;
            useBuffer = false;
        } else {
            actualPad = symmetrizePadding(repaddingOffset);
            repaddedGradShape.height(dataFormat) += repaddingOffset.x;
            repaddedGradShape.width(dataFormat) += repaddingOffset.y;
            useBuffer = true;
        }

        // setup tensors
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, dataFormat);
        cudnn::setTensorDescriptor<T>(gradDesc, repaddedGradShape, dataFormat);

        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,  // OIHW according to the docs
            filterShape[0], filterShape[1], filterShape[2], filterShape[3]));

        // algorithm selection, half float case first
        size_t inputScratchpadSize = 0, kernelScratchpadSize = 0;
        if (cudnn::isHalfFloat<T>()) {
            // fp32 computing might be faster than fp16 computing and is more accurate, so try fp32 computing first
            float kernelGradExecTime, inputGradExecTime = 0;
            setConvDescriptor<float>(convDesc, groups);
            // enable tensor multiplication units (like Tensor Cores)
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
            kernelGradientAlgo = device.selectBackwardFilterAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, kernelGradExecTime, kernelScratchpadSize, kernelMathType);
            if (requireInputGrad)
                inputGradientAlgo = device.selectBackwardDataAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, inputGradExecTime, inputScratchpadSize, inputMathType);

            if (context.isFp16ConvBackwardAllowed()) {
                // fp16 computing is allowed, try it as well
                float fp16KernelGradExecTime, fp16InputGradExecTime = 0;
                size_t fp16KernelScratchpadSize, fp16InputScratchpadSize;
                cudnnMathType_t fp16InputMathType = CUDNN_DEFAULT_MATH, fp16KernelMathType = CUDNN_DEFAULT_MATH;
                cudnnConvolutionBwdFilterAlgo_t fp16KernelGradAlgo;
                cudnnConvolutionBwdDataAlgo_t fp16InputGradAlgo;
                setConvDescriptor<half>(convDesc, groups);
                fp16KernelGradAlgo = device.selectBackwardFilterAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, fp16KernelGradExecTime, fp16KernelScratchpadSize, fp16KernelMathType);
                if (requireInputGrad) {
                    fp16InputGradAlgo = device.selectBackwardDataAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, fp16InputGradExecTime, fp16InputScratchpadSize, fp16InputMathType);
                }

                // compare fp16 and fp32 compute time
                if (fp16KernelGradExecTime + fp16InputGradExecTime < kernelGradExecTime + inputGradExecTime) {
                    // fp16 compute is actually faster than fp32, use it
                    UPSTRIDE_SAYS(context, "fp16 is faster than fp32 for backward pass");
                    kernelScratchpadSize = fp16KernelScratchpadSize;
                    kernelGradientAlgo = fp16KernelGradAlgo;
                    kernelMathType = fp16KernelMathType;
                    if (requireInputGrad) {
                        inputScratchpadSize = fp16InputScratchpadSize;
                        inputGradientAlgo = fp16InputGradAlgo;
                        inputMathType = fp16InputMathType;
                    }
                }
                else
                    setConvDescriptor<float>(convDesc, groups);
            }
        }
        // algorithm selection, other datatypes
        else {
            setConvDescriptor<T>(convDesc, groups);
            float executionTime;
            kernelGradientAlgo = device.selectBackwardFilterAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, executionTime, kernelScratchpadSize, kernelMathType);
            if (requireInputGrad)
                inputGradientAlgo = device.selectBackwardDataAlgo(context, convDesc, inputDesc, gradDesc, filterDesc, executionTime, inputScratchpadSize, inputMathType);
        }

        // allocate scratchpad
        const size_t size = std::max(kernelScratchpadSize, inputScratchpadSize);
        if (scratchpad.getSize() != size)
            scratchpad = cudnn::Memory(size);
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
    void operator()(const Tensor<device::CUDA, const T>& inputTensor,
                    const Tensor<device::CUDA, const T>& kernelTensor,
                    const Tensor<device::CUDA, const T>& gradTensor,
                    Tensor<device::CUDA, T>& kernelGradTensor,
                    Tensor<device::CUDA, T>& inputGradTensor) {
        // pad if needed
        AllocatedTensor<device::CUDA, T>* buffer = nullptr;
        if (useBuffer) {
            bool dirty;
            buffer = &bufferAllocator.get(gradTensor.getDevice(), repaddedGradShape, dirty);
            if (dirty)
                buffer->zero();
            cudnn::insert(gradTensor, *buffer, dataFormat, repaddingOffset);
        }

        // perform the gradient computation
        const float alphaF = 1.0f, betaF = 0.0f;
        const double alphaD = 1.0, betaD = 0.0;

        // set the math type according to the chosen algorithm
        cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, kernelMathType));
        cudnn::Context::raiseIfError(cudnnConvolutionBackwardFilter(
            inputTensor.getDevice().handle(),
            selectScalingParameterPtr<T>(alphaF, alphaD),
            inputDesc, inputTensor.getDataPtr(),
            gradDesc, useBuffer ? buffer->getDataPtr() : gradTensor.getDataPtr(),
            convDesc,
            kernelGradientAlgo,
            scratchpad.pointer(), scratchpad.getSize(),
            selectScalingParameterPtr<T>(betaF, betaD),
            filterDesc, kernelGradTensor.getDataPtr()));

        if (requireInputGrad) {
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, inputMathType));
            cudnn::Context::raiseIfError(cudnnConvolutionBackwardData(
                inputTensor.getDevice().handle(),
                selectScalingParameterPtr<T>(alphaF, alphaD),
                filterDesc, kernelTensor.getDataPtr(),
                gradDesc, useBuffer ? buffer->getDataPtr() : gradTensor.getDataPtr(),
                convDesc,
                inputGradientAlgo,
                scratchpad.pointer(), scratchpad.getSize(),
                selectScalingParameterPtr<T>(betaF, betaD),
                inputDesc, inputGradTensor.getDataPtr()));
        }
    }
};

}  // namespace upstride