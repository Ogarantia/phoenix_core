#pragma once
#include <cuda.h>
#include <cudnn.h>
#include "../backend.hpp"
#include "conv2d_algo_select.hpp"
#include "kernels_utils.hpp"
#include "cublas_v2.h"

namespace upstride {
namespace device {
class CUDA {
   private:
    cudnn::Conv2DAlgorithmSelector conv2dAlgorithms;    //!< runtime conv2d algorithms selector
    cuda::ConvKernelsCache convKernelsCache;            //!< cache for runtime selection of custom CUDA kernels for quaternionic convolutions
    cudaStream_t cudaStream;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    CUDA(const CUDA&) = delete;  // disable copying

   public:

    inline CUDA(const cudaStream_t& stream) : cudaStream(stream) {
        auto status = cudnnCreate(&cudnnHandle);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(std::string("Cannot create cuDNN handle, ") + cudnnGetErrorString(status));
        status = cudnnSetStream(cudnnHandle, cudaStream);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(cudnnGetErrorString(status));

        if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Cannot create cuBLAS handle.");
        if (cublasSetStream(cublasHandle, cudaStream) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Cannot set CUDA stream for cuBLAS.");
    }

    inline ~CUDA() {
        cudnnDestroy(cudnnHandle);
        cublasDestroy(cublasHandle);
    }

    /**
     * @brief Returns CUDA stream associated with the device.
     * Kernels are to be submitted to this stream in order to be executed on the current device.
     * @return const cudaStream_t&
     */
    inline const cudaStream_t& stream() const {
        return cudaStream;
    }

    /**
     * @brief Retrieves cuDNN handle associated with the device.
     * @return const cudnnHandle_t&
     */
    inline const cudnnHandle_t& handle() const {
        return cudnnHandle;
    }

    /**
     * @brief Selects the fastest forward 2D convolution algorithm applicable for a given convolution setting.
     * @param context           A context instance
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param output            The convolution output tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionFwdAlgo_t selectForwardAlgo(const Context& context,
                                                       const cudnnConvolutionDescriptor_t& convDesc,
                                                       const cudnnTensorDescriptor_t& input,
                                                       const cudnnFilterDescriptor_t& kernel,
                                                       const cudnnTensorDescriptor_t& output,
                                                       float& executionTime,
                                                       size_t& scratchpadSize) {
        return conv2dAlgorithms.selectForwardAlgo(context, cudnnHandle, convDesc, input, kernel, output, executionTime, scratchpadSize);
    }

    /**
     * @brief Selects the fastest backward 2D convolution algorithm computing the filter gradient, applicable for
     * a given convolution setting.
     * @param context           A context instance
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionBwdFilterAlgo_t selectBackwardFilterAlgo(const Context& context,
                                                                    const cudnnConvolutionDescriptor_t& convDesc,
                                                                    const cudnnTensorDescriptor_t& input,
                                                                    const cudnnTensorDescriptor_t& grad,
                                                                    const cudnnFilterDescriptor_t& kernel,
                                                                    float& executionTime,
                                                                    size_t& scratchpadSize) {
        return conv2dAlgorithms.selectBackwardFilterAlgo(context, cudnnHandle, convDesc, input, grad, kernel, executionTime, scratchpadSize);
    }

    /**
     * @brief Selects the fastest backward 2D convolution algorithm computing the input (data) gradient, applicable for
     * a given convolution setting.
     * @param context           A context instance
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionBwdDataAlgo_t selectBackwardDataAlgo(const Context& context,
                                                                const cudnnConvolutionDescriptor_t& convDesc,
                                                                const cudnnTensorDescriptor_t& input,
                                                                const cudnnTensorDescriptor_t& grad,
                                                                const cudnnFilterDescriptor_t& kernel,
                                                                float& executionTime,
                                                                size_t& scratchpadSize) {
        return conv2dAlgorithms.selectBackwardDataAlgo(context, cudnnHandle, convDesc, input, grad, kernel, executionTime, scratchpadSize);
    }

    /**
     * @brief Checks cache for a convolution descriptor, sets optimal kernel configuration if found
     * 
     * @param convType          Type of the kernel convolution operation
     * @param convDesc          Descriptor of the kernel convolution
     * @param optimalConf       Parameter used to pass the cached optimal kernel configuration and its profiling record, if found
     * @return                  True if (convType x convDesc) key is found in the cache
     */
    inline bool checkCacheForOptimalKernel(
        const cuda::ConvType convType,
        const cuda::ConvDesc convDesc,
        cuda::PerfResult& optimalConf
    ) {
        return convKernelsCache.checkCache(convType, convDesc, optimalConf);
    }

    /**
     * @brief Adds a kernel configuration and its profiling record to the cache
     * 
     * @param convType          Type of the kernel convolution operation
     * @param convDesc          Descriptor of the kernel convolution
     * @param optimalConf       The optimal kernel configuration and its profiling record
     */
    inline void cacheOptimalKernel(
        const cuda::ConvType convType,
        const cuda::ConvDesc convDesc,
        const cuda::PerfResult& optimalConf
    ) {
        convKernelsCache.addToCache(convType, convDesc, optimalConf);
    }

    /**
     * @brief Retrieves cuBLAS handle associated with the device.
     * @return const cublasHandle_t&
     */
    inline const cublasHandle_t& getCublasHandle() const {
        return cublasHandle;
    }
};
}  // namespace device
}  // namespace upstride