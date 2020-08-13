/**
 * @file conv2d.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief 2D convolution implementation using cuDNN compute backend
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
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
    Shape inputShape, filterShape, outputShape;
    Shape repaddedOutputShape;  //!< intermediate output shape for symmetrically padded input to handle the asymmetric padding
    IntPair padBefore;          //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;           //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;          //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;    //!< offset in the ouptut tensor due to repadding applied to handle the asymmetric padding
    IntPair stride, dilation;
    cudnn::Memory buffer;  //!< buffer in device memory used to store the uncropped output when dealing with the asymmetric padding

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
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void configureBackend(const Shape& inputShape, const Shape& filterShape, const Shape& outputShape, const IntPair& padBefore, const IntPair& padAfter, int groups) {
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
            actualPad = padBefore;
            buffer.free();
        } else {
            actualPad = cudnn::symmetrizePadding(padBefore, padAfter, stride, repaddingOffset);
            repaddedOutputShape.width(dataFormat) += repaddingOffset.x;
            repaddedOutputShape.height(dataFormat) += repaddingOffset.y;

            // allocate intermediate buffer to fit the output into
            buffer = cudnn::Memory(repaddedOutputShape.numel() * sizeof(T));
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
            CUDNN_TENSOR_NCHW,                // OIHW according to the docs
            filterShape[groups > 1 ? 1 : 0],  // FIXME: this inversion is found empirically and is not explained in cuDNN docs; check for regular grouped conv
            filterShape[groups > 1 ? 0 : 1],
            filterShape[2], filterShape[3]));
    }

   public:
    ScalarConv2DFunctor() {
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
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    void configure(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation) {
        this->dataFormat = dataFormat;
        this->stride = stride;
        this->dilation = dilation;
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
                    Tensor<device::CUDA, T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // configure cuDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), filterTensor.getShape(), outputTensor.getShape(), padBefore, padAfter, groups);

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

        // crop, if needed
        if (buffer)
            cudnn::crop(
                Tensor<device::CUDA, const T>(repaddedOutputShape, (const T*)buffer),
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
    Shape inputShape, kernelShape, gradShape;
    Shape repaddedGradShape;  //!< gradient tensor (dy) shape for symmetrically padded input to handle the asymmetric padding
    IntPair padBefore;        //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;         //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair actualPad;        //!< zero padding actually applied by cuDNN (both at the beginning and at the end of every spatial dimension)
    IntPair repaddingOffset;  //!< offset in the ouptut tensor due to repadding applied to handle the asymmetric padding
    IntPair stride, dilation;
    DataFormat dataFormat;
    cudnn::Memory buffer;  //!< intermediate buffer to store repadded output gradient

    cudnnConvolutionDescriptor_t convDesc;   //!< convolution operation descriptor, as in forward
    cudnnTensorDescriptor_t inputDesc;       //!< input (x) descriptor
    cudnnTensorDescriptor_t gradDesc;        //!< output value gradient (dy) descriptor (which is an input of this operation)
    cudnnFilterDescriptor_t kernelGradDesc;  //!< kernel gradient (dw) descriptor (output of this operation)

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param kernelShape       kernel tensor shape
     * @param gradShape         grad tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    void configureBackend(const Shape& inputShape,
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

        repaddedGradShape = gradShape;

        // check for padding symmetry
        if (padBefore == padAfter) {
            actualPad = padBefore;
            buffer.free();
        } else {
            actualPad = cudnn::symmetrizePadding(padBefore, padAfter, stride, repaddingOffset);
            repaddedGradShape.width(dataFormat) += repaddingOffset.x;
            repaddedGradShape.height(dataFormat) += repaddingOffset.y;

            // allocate the intermediate buffer
            buffer = cudnn::Memory(repaddedGradShape.numel() * sizeof(T));
            buffer.zero();
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
            kernelGradDesc,
            cudnn::getDataType<T>(),
            CUDNN_TENSOR_NCHW,                // OIHW according to the docs
            kernelShape[groups > 1 ? 1 : 0],  // FIXME: this inversion is found empirically and is not explained in cuDNN docs; check for regular grouped conv
            kernelShape[groups > 1 ? 0 : 1],
            kernelShape[2], kernelShape[3]));
    }

   public:
    ScalarConv2DGradFunctor() {
        cudnn::Context::raiseIfError(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&inputDesc));
        cudnn::Context::raiseIfError(cudnnCreateTensorDescriptor(&gradDesc));
        cudnn::Context::raiseIfError(cudnnCreateFilterDescriptor(&kernelGradDesc));
    }

    ~ScalarConv2DGradFunctor() {
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(gradDesc);
        cudnnDestroyFilterDescriptor(kernelGradDesc);
    }

    void configure(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) {
        if (requireInputGrad)
            throw std::runtime_error("Conv2D gradient computation with respect to the input tensor is not implemented");
        this->dataFormat = dataFormat;
        this->stride = stride;
        this->dilation = dilation;
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
                    Tensor<device::CUDA, T>& inputGradTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter,
                    int groups = 1) {
        // configure oneDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), kernelTensor.getShape(), gradTensor.getShape(), padBefore, padAfter, groups);

        // pad if needed
        if (buffer) {
            Tensor<device::CUDA, T> repaddedGradTensor(repaddedGradShape, (T*)buffer);
            cudnn::insert(gradTensor, repaddedGradTensor, dataFormat, repaddingOffset);
        }

        // perform the gradient computation
        const T alpha = 1, beta = 0;
        cudnn::Context::raiseIfError(cudnnConvolutionBackwardFilter(
            cudnn::Context::getInstance().getHandle(),
            &alpha,
            inputDesc, inputTensor.getDataPtr(),
            gradDesc, buffer ? buffer : gradTensor.getDataPtr(),
            convDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            nullptr, 0,
            &beta,
            kernelGradDesc, kernelGradTensor.getDataPtr()));
    }
};

}  // namespace upstride