#pragma once
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "conv2d_algo_select.hpp"
#include "kernels_utils.hpp"
#include "../backend.hpp"
#include "../device.hpp"
#include "../../isolated_thread.hpp"

namespace upstride {
namespace device {
class CUDA : public Device {
   private:
    cudnn::Conv2DAlgorithmSelector conv2dAlgorithms;    //!< runtime conv2d algorithms selector
    cuda::ConvKernelsCache convKernelsCache;            //!< cache for runtime selection of custom CUDA kernels for quaternionic convolutions
    IsolatedThread allocator;                           //!< thread performing memory allocations
    cudaStream_t cudaStream;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    int registersPerThreadBlock;                        //!< maximum number of registers per thread block
    size_t alignmentConstraint;                         //!< number of bytes used to aligned pointers for this specific device

    CUDA(const CUDA&) = delete;  // disable copying

    /**
     * @brief Attaches a specific CUDA device to the current thread.
     * @param device    The CUDA device number
     */
    void attachDevice(int device);

    /**
     * @brief Allocates GPU memory.
     * Wraps a call to a low-level CUDA API. Called from within the allocator thread. Not to be called directly.
     * @param size      Memory size in bytes
     * @param memory    Address of the allocated buffer
     */
    void internalMalloc(size_t size, void* &memory);

    /**
     * @brief Frees an allocated buffer on GPU.
     * Wraps a call to a low-level CUDA API. Called from within the allocator thread. Not to be called directly.
     * @param memory    Address of the allocated buffer
     */
    void internalFree(void* memory);

   public:
    CUDA(Context& context, const cudaStream_t& stream);

    inline ~CUDA() {
        freeWorkspaceMemory();
        cudnnDestroy(cudnnHandle);
        cublasDestroy(cublasHandle);
    }

    inline size_t getAlignmentConstraint() const override {
        return alignmentConstraint;
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
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param output            The convolution output tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionFwdAlgo_t selectForwardAlgo(const cudnnConvolutionDescriptor_t& convDesc,
                                                       const cudnnTensorDescriptor_t& input,
                                                       const cudnnFilterDescriptor_t& kernel,
                                                       const cudnnTensorDescriptor_t& output,
                                                       float& executionTime,
                                                       size_t& scratchpadSize,
                                                       cudnnMathType_t& mathType) {
        return conv2dAlgorithms.selectForwardAlgo(cudnnHandle, convDesc, input, kernel, output, executionTime, scratchpadSize, mathType);
    }

    /**
     * @brief Selects the fastest backward 2D convolution algorithm computing the filter gradient, applicable for
     * a given convolution setting.
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionBwdFilterAlgo_t selectBackwardFilterAlgo(const cudnnConvolutionDescriptor_t& convDesc,
                                                                    const cudnnTensorDescriptor_t& input,
                                                                    const cudnnTensorDescriptor_t& grad,
                                                                    const cudnnFilterDescriptor_t& kernel,
                                                                    float& executionTime,
                                                                    size_t& scratchpadSize,
                                                                    cudnnMathType_t& mathType) {
        return conv2dAlgorithms.selectBackwardFilterAlgo(cudnnHandle, convDesc, input, grad, kernel, executionTime, scratchpadSize, mathType);
    }

    /**
     * @brief Selects the fastest backward 2D convolution algorithm computing the input (data) gradient, applicable for
     * a given convolution setting.
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param time              Returns execution time in milliseconds taken by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    inline cudnnConvolutionBwdDataAlgo_t selectBackwardDataAlgo(const cudnnConvolutionDescriptor_t& convDesc,
                                                                const cudnnTensorDescriptor_t& input,
                                                                const cudnnTensorDescriptor_t& grad,
                                                                const cudnnFilterDescriptor_t& kernel,
                                                                float& executionTime,
                                                                size_t& scratchpadSize,
                                                                cudnnMathType_t& mathType) {
        return conv2dAlgorithms.selectBackwardDataAlgo(cudnnHandle, convDesc, input, grad, kernel, executionTime, scratchpadSize, mathType);
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

    /**
     * @brief Get the number of registers available per thread block for either the current device or across all devices
     */
    inline int getRegistersPerThreadBlock() const { return registersPerThreadBlock; }

    /**
     * @brief Allocates GPU memory.
     * This function is thread-safe. The allocated buffer can be freed in another thread with free() function.
     * @param size      Size in bytes.
     * @return the allocated memory address.
     */
    void* malloc(size_t size) override;

    template<typename T>
    inline T* malloc(size_t size) {
        return static_cast<T*>(malloc(size));
    }

    /**
     * @brief Frees GPU memory.
     * This function is thread-safe. It can recycle a memory buffer allocated from a different thread.
     * @param memory    Address of the buffer to free
     */
    void free(void* memory) override;
};
}  // namespace device
}  // namespace upstride