#include "device.hpp"
#include "context.hpp"


namespace upstride {
namespace device {

// GPU querying

/**
 * @brief Get the number of registers available per thread block on the specified device
 *
 * @param dev                               the device to be queried for the available registers
 */
int getDeviceRegistersPerThreadBlock(int dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudnn::Context::raiseIfError("getDeviceRegistersPerThreadBlock cudaGetDeviceProperties failed");
    return deviceProp.regsPerBlock;
}


void CUDA::attachDevice(int device) {
    cudaSetDevice(device);
}


void CUDA::internalMalloc(size_t size, void* &memory) {
    cudnn::Context::raiseIfError(cudaMalloc(&memory, size));
}


void CUDA::internalFree(void* memory) {
    cudnn::Context::raiseIfError(cudaFree(memory));
}


CUDA::CUDA(const cudaStream_t& stream) : cudaStream(stream) {
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

    // attach the allocator thread to the current CUDA device
    int currentDevice;
    if (cudaGetDevice(&currentDevice) != cudaError::cudaSuccess)
        throw std::runtime_error("cudaGetDevice failed");
    allocator.call(this, &CUDA::attachDevice, currentDevice);

    // Query number of registers per thread block
    registersPerThreadBlock = getDeviceRegistersPerThreadBlock(currentDevice);
}


void CUDA::free(void* memory) {
    allocator.call(this, &CUDA::internalFree, memory);
}

}
}