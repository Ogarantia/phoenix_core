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

namespace cudnn {
/**
 * @brief For an asymmetric input padding, computes symmetric padding allowing to have the same output tensor as for the original asymmetric padding, up to a crop. Otherwise transmits the input padding as is.
 *
 * @param padBefore         padding at the beginning of spatial dimensions
 * @param padAfter          padding at thg end of spatial dimensions
 * @param stride            stride
 * @param repaddingOffset   the offset value of the output after the repadding
 * @return the symmetric padding.
 */
inline IntPair symmetrizePadding(const IntPair& padBefore, const IntPair& padAfter, const IntPair& stride, IntPair& repaddingOffset) {
    // Proceed with the symmetric padding covering the requested padding.
    // Adding one step (stride) to padBefore is equivalent to add an entry to output at the beginning of every dimension.
    // This is be cropped further on after the convolution is computed.
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
}  // namespace cudnn

/**
 * @brief 2D convolution implementation using cuDNN.
 * @tparam T    A scalar datatype for the tensor content
 */
template <typename T>
class ScalarConv2DFunctor<device::CUDA, T> {
   private:
    const IntPair stride, dilation;
    const DataFormat dataFormat;

    Shape inputShape, filterShape, outputShape;
    IntPair padBefore;        //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;         //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;        //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;  //!< offset in the ouptut tensor due to repadding applied to handle the asymmetric padding
    Shape repaddedOutputShape;
    bool useBuffer;                                      //!< if true, an intermediate buffer is used to store repadded input tensor
    DeferredAllocator<device::CUDA, T> bufferAllocator;  //!< the intermediate buffer to store the repadded input tensor

    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;

   public:
    /**
     * @brief Sets main convolution parameters independent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    ScalarConv2DFunctor(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation) : dataFormat(dataFormat),
                                                                                                 stride(stride),
                                                                                                 dilation(dilation) {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&outputDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&filterDesc));
    }

    ~ScalarConv2DFunctor() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
    }

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     */
    void configure(const Shape& inputShape, const Shape& filterShape, const Shape& outputShape, const IntPair& padBefore, const IntPair& padAfter, int groups) {
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
            actualPad = cudnn::symmetrizePadding(padBefore, padAfter, stride, repaddingOffset);
            repaddedOutputShape.width(dataFormat) += repaddingOffset.x;
            repaddedOutputShape.height(dataFormat) += repaddingOffset.y;
            useBuffer = true;
        }

        // setup convolution descriptor
        cudnn::Context::raiseIfError(cudnnSetConvolution2dDescriptor(
            convDesc,
            actualPad.y, actualPad.x,
            stride.y, stride.x,
            dilation.y, dilation.x,
            CUDNN_CROSS_CORRELATION,
            cudnn::getDataType<T>()));

        // enable groups
        if (groups > 1)
            cudnn::Context::raiseIfError(cudnnSetConvolutionGroupCount(convDesc, groups));

        // setup tensors
        cudnn::setTensorDescriptor<T>(outputDesc, repaddedOutputShape, dataFormat);
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, dataFormat);

        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,  // OIHW according to the docs
            filterShape[0], filterShape[1], filterShape[2], filterShape[3]));
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param outputTensor      Output tensor
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void operator()(const Tensor<device::CUDA, const T>& inputTensor,
                    const Tensor<device::CUDA, const T>& filterTensor,
                    Tensor<device::CUDA, T>& outputTensor) {
        // allocate buffer, if needed
        AllocatedTensor<device::CUDA, T>* buffer = nullptr;
        if (useBuffer)
            buffer = &bufferAllocator.get(outputTensor.getDevice(), repaddedOutputShape);

        // perform the convolution
        const T alpha = 1, beta = 0;
        cudnn::Context::raiseIfError(cudnnConvolutionForward(
            inputTensor.getDevice().handle(),
            &alpha,
            inputDesc, inputTensor.getDataPtr(),
            filterDesc, filterTensor.getDataPtr(),
            convDesc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            nullptr, 0,
            &beta,
            outputDesc, useBuffer ? buffer->getDataPtr() : outputTensor.getDataPtr()));

        // crop, if needed
        if (useBuffer)
            cudnn::crop(
                *buffer,
                outputTensor,
                dataFormat,
                repaddingOffset);
    }
};

/**
 * @brief 2D backward convolution implementation using cuDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class ScalarConv2DGradFunctor<device::CUDA, T> {
   private:
    const DataFormat dataFormat;
    const IntPair stride, dilation;
    const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not
    Shape inputShape, kernelShape, gradShape;
    IntPair padBefore;        //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;         //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;        //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;  //!< offset in the ouptut tensor due to repadding applied to handle the asymmetric padding
    Shape repaddedGradShape;
    bool useBuffer;                                      //!< if true, an intermediate buffer is used to repad the gradient tensor
    DeferredAllocator<device::CUDA, T> bufferAllocator;  //!< the intermediate buffer to store the repadded gradient tensor

    cudnnConvolutionDescriptor_t convDesc;  //!< convolution operation descriptor, as in forward
    cudnnTensorDescriptor_t inputDesc;      //!< input (x) descriptor
    cudnnTensorDescriptor_t gradDesc;       //!< output value gradient (dy) descriptor (which is an input of this operation)
    cudnnFilterDescriptor_t kernelDesc;     //!< kernel gradient (dw) descriptor (output of this operation)

   public:
    ScalarConv2DGradFunctor(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) : dataFormat(dataFormat),
                                                                                                                            stride(stride),
                                                                                                                            dilation(dilation),
                                                                                                                            requireInputGrad(requireInputGrad) {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&gradDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&kernelDesc));
    }

    ~ScalarConv2DGradFunctor() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(gradDesc);
        cudnnDestroyFilterDescriptor(kernelDesc);
    }

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param kernelShape       kernel tensor shape
     * @param gradShape         grad tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void configure(const Shape& inputShape,
                   const Shape& kernelShape,
                   const Shape& gradShape,
                   const IntPair& padBefore,
                   const IntPair& padAfter,
                   int groups) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->kernelShape == kernelShape && this->gradShape == gradShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->kernelShape = kernelShape;
        this->gradShape = gradShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // check for padding symmetry
        repaddedGradShape = gradShape;
        if (padBefore == padAfter) {
            actualPad = padBefore;
            useBuffer = false;
        } else {
            actualPad = cudnn::symmetrizePadding(padBefore, padAfter, stride, repaddingOffset);
            repaddedGradShape.width(dataFormat) += repaddingOffset.x;
            repaddedGradShape.height(dataFormat) += repaddingOffset.y;
            useBuffer = true;
        }

        // setup convolution descriptor
        cudnn::Context::raiseIfError(cudnnSetConvolution2dDescriptor(
            convDesc,
            actualPad.y, actualPad.x,
            stride.y, stride.x,
            dilation.y, dilation.x,
            CUDNN_CROSS_CORRELATION,
            cudnn::getDataType<T>()));

        // enable groups
        if (groups > 1)
            cudnn::Context::raiseIfError(cudnnSetConvolutionGroupCount(convDesc, groups));

        // setup tensors
        cudnn::setTensorDescriptor<T>(inputDesc, inputShape, dataFormat);
        cudnn::setTensorDescriptor<T>(gradDesc, repaddedGradShape, dataFormat);

        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            kernelDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,  // OIHW according to the docs
            kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3]));
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
        const T alpha = 1, beta = 0;
        cudnn::Context::raiseIfError(cudnnConvolutionBackwardFilter(
            inputTensor.getDevice().handle(),
            &alpha,
            inputDesc, inputTensor.getDataPtr(),
            gradDesc, useBuffer ? buffer->getDataPtr() : gradTensor.getDataPtr(),
            convDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            nullptr, 0,
            &beta,
            kernelDesc, kernelGradTensor.getDataPtr()));

        if (requireInputGrad) {
            cudnn::Context::raiseIfError(cudnnConvolutionBackwardData(
                inputTensor.getDevice().handle(),
                &alpha,
                kernelDesc, kernelTensor.getDataPtr(),
                gradDesc, useBuffer ? buffer->getDataPtr() : gradTensor.getDataPtr(),
                convDesc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                nullptr, 0,
                &beta,
                inputDesc, inputGradTensor.getDataPtr()));
        }
    }
};

}  // namespace upstride