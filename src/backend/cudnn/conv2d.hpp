/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementation using cuDNN compute backend
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include "../memory_request.hpp"
#include "../temporary_tensor.hpp"
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
template <typename T>
class ScalarConv2DBase {
   protected:
    device::CUDA& device;
    const DataFormat tensorFormat;
    const IntPair stride, dilation;
    IntPair actualPad;                          //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;                    //!< offset in the output tensor due to the additional padding applied to handle the asymmetric padding
    Shape repaddedOutputShape;                  //!< shape of the forward output / backward gradient tensor having an additional symmetrized zero padding
    bool useBuffer;                             //!< if true, an intermediate buffer is used to store repadded input tensor
    size_t scratchpadSize;                      //!< memory size of cuDNN workspace buffer
    Pointer scratchpad;                         //!< pointer to the cuDNN workspace buffer
    TemporaryTensor<device::CUDA, T> buffer;    //!< intermediate buffer to store the repadded input tensor

    cudnnConvolutionDescriptor_t convDesc;      //!< cuDNN convolution operator descriptor
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;

    ScalarConv2DBase(
        device::CUDA& device,
        DataFormat tensorFormat,
        FilterLayout filterLayout,
        const IntPair& stride,
        const IntPair& dilation,
        const Shape& filterShape,
        const Shape& outputShape,
        const IntPair& padBefore,
        const IntPair& padAfter
    ):
        device(device), tensorFormat(tensorFormat), stride(stride), dilation(dilation), scratchpadSize(0), buffer(device)
    {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&outputDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&filterDesc));

        // map kernel layout to cuDNN tensor format, https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilter4dDescriptor
        cudnnTensorFormat_t filterTensorFormat;
        if (filterLayout == FilterLayout::OIHW)
            filterTensorFormat = CUDNN_TENSOR_NCHW;
        else if (filterLayout == FilterLayout::OHWI)
            filterTensorFormat = CUDNN_TENSOR_NHWC;
        else
            throw std::invalid_argument("Unsupported conv2D filter layout");

        // set filter descriptor
        Conv2DFilterLayout filter(filterLayout);
        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnn::getDataType<T>(),
            filterTensorFormat,
            filter.numOutputChannels(filterShape), filter.numInputChannels(filterShape), filter.height(filterShape), filter.width(filterShape)));

        // check for padding symmetry
        repaddedOutputShape = outputShape;
        if (padBefore == padAfter) {
            actualPad = padBefore;
            useBuffer = false;
        } else {
            actualPad = symmetrizePadding(padBefore, padAfter, repaddingOffset);
            repaddedOutputShape.height(tensorFormat) += repaddingOffset.x;
            repaddedOutputShape.width(tensorFormat) += repaddingOffset.y;
            useBuffer = true;
        }

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
     * @param padBefore         top/left zero padding
     * @param padAfter          bottom/right zero padding
     * @param repaddingOffset   the offset value of the output after the repadding
     * @return the symmetric padding.
     */
    inline IntPair symmetrizePadding(const IntPair& padBefore, const IntPair& padAfter, IntPair& repaddingOffset) {
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

   public:
    /**
     * @brief Prepares intermediate memory required by the operation.
     * @param memory        A memory request instance to allocate the intermediate buffers
     */
    inline void prepare(MemoryRequest& memory) {
        scratchpad = memory.alloc(scratchpadSize);
        if (useBuffer)
            buffer = TemporaryTensor<device::CUDA, T>(device, memory, repaddedOutputShape);
    }
};


/**
 * @brief 2D convolution implementation using cuDNN.
 * @tparam T    A scalar datatype of tensors content
 */
template <typename T>
class ScalarConv2DFunctor<device::CUDA, T> : public ScalarConv2DBase<T> {
    using Base = ScalarConv2DBase<T>;
   private:
    cudnnConvolutionFwdAlgo_t algorithm;        //!< cuDNN convolution computation algorithm
    using Base::inputDesc;
    using Base::outputDesc;
    using Base::filterDesc;
    using Base::convDesc;
    using Base::repaddedOutputShape;
    using Base::repaddingOffset;
    using Base::tensorFormat;
    using Base::scratchpadSize;
    using Base::scratchpad;
    using Base::useBuffer;
    using Base::buffer;

   public:
    /**
     * @brief Instantiates a Conv2D operation.
     * @param device            A device the operation will be executed on
     * @param tensorFormat      Memory layout of input and output tensors
     * @param filterLayout      Convolution filter layout
     * @param stride            Convolution stride
     * @param dilation          Convolution dilation
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param biasShape         Bias tensor shape (empty if no bias addition is enabled)
     * @param outputShape       Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    ScalarConv2DFunctor(
        device::CUDA& device,
        DataFormat tensorFormat,
        FilterLayout filterLayout,
        const IntPair& stride,
        const IntPair& dilation,
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& biasShape,
        const Shape& outputShape,
        const IntPair& padBefore,
        const IntPair& padAfter,
        int groups
    ):
        ScalarConv2DBase<T>(device, tensorFormat, filterLayout, stride, dilation, filterShape, outputShape, padBefore, padAfter)
    {
        // setup tensors
        cudnn::setTensorDescriptor<T>(outputDesc, repaddedOutputShape, tensorFormat);
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, tensorFormat);

        // algorithm selection, half float case first
        const cudnnTensorFormat_t cudnnTensorFmt = cudnn::dataFormatToTensorFormat(tensorFormat);
        float executionTime;
        cudnnMathType_t mathType;
        if (cudnn::isHalfFloat<T>()) {
            // fp32 computing might be faster than fp16 computing and is more accurate, so try fp32 computing first
            Base::template setConvDescriptor<float>(convDesc, groups);
            // enable tensor multiplication units (like Tensor Cores)
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
            algorithm = device.selectForwardAlgo(convDesc, inputDesc, filterDesc, outputDesc, cudnnTensorFmt, executionTime, scratchpadSize, mathType);

            if (device.getContext().isFp16ConvForwardAllowed()) {
                // fp16 computing is allowed, try it as well
                float fp16ExecTime;
                size_t fp16ScratchpadSize;
                cudnnMathType_t fp16MathType;
                Base::template setConvDescriptor<half>(convDesc, groups);
                cudnnConvolutionFwdAlgo_t fp16Algorithm = device.selectForwardAlgo(convDesc, inputDesc, filterDesc, outputDesc, cudnnTensorFmt, fp16ExecTime, fp16ScratchpadSize, fp16MathType);
                if (fp16ExecTime < executionTime) {
                    // fp16 compute is actually faster than fp32, use it
                    UPSTRIDE_SAYS("fp16 is faster than fp32 for forward pass");
                    scratchpadSize = fp16ScratchpadSize;
                    algorithm = fp16Algorithm;
                    mathType = fp16MathType;
                }
                else
                    Base::template setConvDescriptor<float>(convDesc, groups);
            }
        }
        // algorithm selection for other datatypes
        else {
            Base::template setConvDescriptor<T>(convDesc, groups);
            algorithm = device.selectForwardAlgo(convDesc, inputDesc, filterDesc, outputDesc, cudnnTensorFmt, executionTime, scratchpadSize, mathType);
        }

        // set the math type according to the chosen algorithm
        cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, mathType));
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
        if (useBuffer)
            Base::buffer.prepare();

        // perform the convolution
        const float alphaF = 1.0f, betaF = 0.0f;
        const double alphaD = 1.0, betaD = 0.0;
        cudnn::Context::raiseIfError(cudnnConvolutionForward(
            Base::device.handle(),
            selectScalingParameterPtr<T>(alphaF, alphaD),
            inputDesc, inputTensor.getDataPtr(),
            filterDesc, filterTensor.getDataPtr(),
            convDesc,
            algorithm,
            scratchpad, scratchpadSize,
            selectScalingParameterPtr<T>(betaF, betaD),
            outputDesc, useBuffer ? buffer.getDataPtr() : outputTensor.getDataPtr()));

        // crop, if needed
        if (useBuffer)
            cudnn::crop(buffer, outputTensor, tensorFormat, repaddingOffset);

        // add bias
        if (biasTensor)
            cudnn::addBias(outputTensor, *biasTensor, tensorFormat);

    }
};  // namespace upstride

/**
 * @brief 2D backward convolution implementation using cuDNN
 * @tparam T    A scalar datatype of tensors content
 */
template <typename T>
class ScalarConv2DGradFunctor<device::CUDA, T> : public ScalarConv2DBase<T> {
    using Base = ScalarConv2DBase<T>;
   private:
    const bool requireInputGrad;                        //!< Used to determine if inputGrad needs to be computed or not
    cudnnConvolutionBwdDataAlgo_t inputGradientAlgo;    //!< cuDNN backward algorithm used to compute the input (data) gradient
    cudnnConvolutionBwdFilterAlgo_t kernelGradientAlgo; //!< cuDNN backward algorithm used to compute the kernel (filter) gradient
    cudnnTensorDescriptor_t gradDesc;                   //!< output value gradient (dy) descriptor (which is an input of this operation)
    cudnnMathType_t inputMathType, kernelMathType;      //!< math types for the chosen algorithms, either regular math or Tensor Cores
    using Base::inputDesc;
    using Base::outputDesc;
    using Base::filterDesc;
    using Base::convDesc;
    using Base::repaddedOutputShape;
    using Base::repaddingOffset;
    using Base::tensorFormat;
    using Base::scratchpadSize;
    using Base::scratchpad;
    using Base::useBuffer;
    using Base::buffer;

   public:
    /**
     * @brief Instantiates a Conv2D backward operation.
     * @param device            A device the operation will be executed on
     * @param tensorFormat      Memory layout of input and output tensors
     * @param filterLayout      Convolution filter layout
     * @param stride            Convolution stride
     * @param dilation          Convolution dilation
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputShape       Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed
     */
    ScalarConv2DGradFunctor(
        device::CUDA& device,
        DataFormat tensorFormat,
        FilterLayout filterLayout,
        const IntPair& stride,
        const IntPair& dilation,
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& outputShape,
        const IntPair& padBefore,
        const IntPair& padAfter,
        int groups,
        bool requireInputGrad
    ):
        ScalarConv2DBase<T>(device, tensorFormat, filterLayout, stride, dilation, filterShape, outputShape, padBefore, padAfter),
        requireInputGrad(requireInputGrad)
    {
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&gradDesc));

        // setup tensors
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, tensorFormat);
        cudnn::setTensorDescriptor<T>(gradDesc, repaddedOutputShape, tensorFormat);

        // algorithm selection, half float case first
        const cudnnTensorFormat_t cudnnTensorFmt = cudnn::dataFormatToTensorFormat(tensorFormat);
        size_t inputScratchpadSize = 0, kernelScratchpadSize = 0;
        if (cudnn::isHalfFloat<T>()) {
            // fp32 computing might be faster than fp16 computing and is more accurate, so try fp32 computing first
            float kernelGradExecTime, inputGradExecTime = 0;
            Base::template setConvDescriptor<float>(convDesc, groups);
            // enable tensor multiplication units (like Tensor Cores)
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
            kernelGradientAlgo = device.selectBackwardFilterAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, kernelGradExecTime, kernelScratchpadSize, kernelMathType);
            if (requireInputGrad)
                inputGradientAlgo = device.selectBackwardDataAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, inputGradExecTime, inputScratchpadSize, inputMathType);

            if (device.getContext().isFp16ConvBackwardAllowed()) {
                // fp16 computing is allowed, try it as well
                float fp16KernelGradExecTime, fp16InputGradExecTime = 0;
                size_t fp16KernelScratchpadSize, fp16InputScratchpadSize;
                cudnnMathType_t fp16InputMathType = CUDNN_DEFAULT_MATH, fp16KernelMathType = CUDNN_DEFAULT_MATH;
                cudnnConvolutionBwdFilterAlgo_t fp16KernelGradAlgo;
                cudnnConvolutionBwdDataAlgo_t fp16InputGradAlgo;
                Base::template setConvDescriptor<half>(convDesc, groups);
                fp16KernelGradAlgo = device.selectBackwardFilterAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, fp16KernelGradExecTime, fp16KernelScratchpadSize, fp16KernelMathType);
                if (requireInputGrad) {
                    fp16InputGradAlgo = device.selectBackwardDataAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, fp16InputGradExecTime, fp16InputScratchpadSize, fp16InputMathType);
                }

                // compare fp16 and fp32 compute time
                if (fp16KernelGradExecTime + fp16InputGradExecTime < kernelGradExecTime + inputGradExecTime) {
                    // fp16 compute is actually faster than fp32, use it
                    UPSTRIDE_SAYS("fp16 is faster than fp32 for backward pass");
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
                    Base::template setConvDescriptor<float>(convDesc, groups);
            }
        }
        // algorithm selection, other datatypes
        else {
            Base::template setConvDescriptor<T>(convDesc, groups);
            float executionTime;
            kernelGradientAlgo = device.selectBackwardFilterAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, executionTime, kernelScratchpadSize, kernelMathType);
            if (requireInputGrad)
                inputGradientAlgo = device.selectBackwardDataAlgo(convDesc, inputDesc, gradDesc, filterDesc, cudnnTensorFmt, executionTime, inputScratchpadSize, inputMathType);
        }

        // set scratchpad size as max
        scratchpadSize = std::max(kernelScratchpadSize, inputScratchpadSize);
    }

    ~ScalarConv2DGradFunctor() {
        cudnnDestroyTensorDescriptor(gradDesc);
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
        if (useBuffer) {
            buffer.prepare();
            cudnn::insert(gradTensor, buffer, tensorFormat, repaddingOffset);
        }

        // perform the gradient computation
        const float alphaF = 1.0f, betaF = 0.0f;
        const double alphaD = 1.0, betaD = 0.0;

        // set the math type according to the chosen algorithm
        cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, kernelMathType));
        cudnn::Context::raiseIfError(cudnnConvolutionBackwardFilter(
            Base::device.handle(),
            selectScalingParameterPtr<T>(alphaF, alphaD),
            inputDesc, inputTensor.getDataPtr(),
            gradDesc, useBuffer ? buffer.getDataPtr() : gradTensor.getDataPtr(),
            convDesc,
            kernelGradientAlgo,
            scratchpad, scratchpadSize,
            selectScalingParameterPtr<T>(betaF, betaD),
            filterDesc, kernelGradTensor.getDataPtr()));

        if (requireInputGrad) {
            cudnn::Context::raiseIfError(cudnnSetConvolutionMathType(convDesc, inputMathType));
            cudnn::Context::raiseIfError(cudnnConvolutionBackwardData(
                Base::device.handle(),
                selectScalingParameterPtr<T>(alphaF, alphaD),
                filterDesc, kernelTensor.getDataPtr(),
                gradDesc, useBuffer ? buffer.getDataPtr() : gradTensor.getDataPtr(),
                convDesc,
                inputGradientAlgo,
                scratchpad, scratchpadSize,
                selectScalingParameterPtr<T>(betaF, betaD),
                inputDesc, inputGradTensor.getDataPtr()));
        }
    }
};

}  // namespace upstride