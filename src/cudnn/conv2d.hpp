/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementation using cuDNN compute backend
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include "../utils.hpp"
#include "context.hpp"
#include "kernels.hpp"

namespace upstride {

/**
 * @brief Regular 2D convolution implementation using cuDNN.
 * @tparam T    A scalar datatype for the tensor content
 */
template <typename T>
class UpstrideConv2DFunctor<upstride::device::GPU, T> {
   private:
    Shape inputShape, filterShape, outputShape;
    Shape repaddedOutputShape;  //!< intermediate output shape for symmetrically padded input to handle the asymmetric padding
    IntPair padBefore;          //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;           //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;          //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;    //!< offset in the ouptut tensor due to repadding applied to handle the asymmetric padding
    IntPair stride, dilation;
    void* buffer;  //!< buffer in device memory used to store the uncropped output when dealing with the asymmetric padding

    DataFormat dataFormat;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     */
    void configureBackend(const Shape& inputShape, const Shape& filterShape, const Shape& outputShape, const IntPair& padBefore, const IntPair& padAfter) {
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

        repaddedOutputShape = outputShape;

        // check for padding symmetry
        if (padBefore == padAfter) {
            if (buffer) {
                cudaFree(buffer);
                buffer = nullptr;
            }
            actualPad = padBefore;
        } else {
            // Proceed with the symmetric padding covering the requested padding.
            // Adding one step (stride) to padBefore is equivalent to add an entry to output at the beginning of every dimension.
            // This is be cropped further on after the convolution is computed.
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

            repaddedOutputShape.width(dataFormat) += repaddingOffset.x;
            repaddedOutputShape.height(dataFormat) += repaddingOffset.y;

            // allocate intermediate buffer to fit the output into
            if (buffer)
                cudaFree(buffer);
            cudnn::Context::raiseIfError(cudaMalloc(&buffer, repaddedOutputShape.numel() * sizeof(T)));
        }

        // setup convolution descriptor
        cudnn::Context::raiseIfError(cudnnSetConvolution2dDescriptor(
            convDesc,
            actualPad.y, actualPad.x,
            stride.y, stride.x,
            dilation.y, dilation.x,
            CUDNN_CROSS_CORRELATION,
            cudnn::getDataType<T>()));

        // setup tensors
        cudnn::Context::raiseIfError(cudnnSetTensor4dDescriptor(
            outputDesc,
            cudnn::dataFormatToTensorFormat(dataFormat),
            cudnn::getDataType<T>(),
            repaddedOutputShape[0], repaddedOutputShape.depth(dataFormat), repaddedOutputShape.height(dataFormat), repaddedOutputShape.width(dataFormat)));

        cudnn::Context::raiseIfError(cudnnSetTensor4dDescriptor(
            inputDesc,
            cudnn::dataFormatToTensorFormat(dataFormat),
            cudnn::getDataType<T>(),
            inputShape[0], inputShape.depth(dataFormat), inputShape.height(dataFormat), inputShape.width(dataFormat)));

        cudnn::Context::raiseIfError(cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,  // OIHW according to the docs
            filterShape[0], filterShape[1], filterShape[2], filterShape[3]));
    }

   public:
    UpstrideConv2DFunctor() : buffer(nullptr) {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&outputDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&filterDesc));
    }

    ~UpstrideConv2DFunctor() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudaFree(buffer);
    }

    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    void configure(DataFormat dataFormat, const IntTuple& stride, const IntTuple& dilation) {
        this->dataFormat = dataFormat;
        getSpatialStep(stride, 1, this->stride);
        getSpatialStep(dilation, 1, this->dilation);
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& filterTensor,
                    Tensor<T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter) {
        // configure cuDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), filterTensor.getShape(), outputTensor.getShape(), padBefore, padAfter);

        // perform the convolution
        const T alpha = 1, beta = 0;
        cudnn::Context::raiseIfError(cudnnConvolutionForward(
            cudnn::Context::getInstance().getHandle(),
            &alpha,
            inputDesc, inputTensor.getDataPtr(),
            filterDesc, filterTensor.getDataPtr(),
            convDesc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            nullptr, 0,
            &beta,
            outputDesc, buffer ? buffer : outputTensor.getDataPtr()));

        // crop
        if (buffer)
            cudnn::crop(
                Tensor<const T>(repaddedOutputShape, (const T*)buffer),
                outputTensor,
                dataFormat,
                repaddingOffset);
    }
};

}  // namespace upstride