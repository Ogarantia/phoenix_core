#include "device.hpp"
#include "context.hpp"


namespace upstride {
namespace device {


void CUDA::attachDevice(int device) {
    cudaSetDevice(device);
}


void CUDA::internalMalloc(size_t size, void* &memory) {
    cudnn::Context::raiseIfError(cudaMalloc(&memory, size));
}


void CUDA::internalFree(void* memory) {
    cudnn::Context::raiseIfError(cudaFree(memory));
}


CUDA::CUDA(Context& context, const cudaStream_t& stream) : Device(context), cudaStream(stream), bypassCudnnHandleDestruction(true) {
    // create cuDNN handle
    auto status = cudnnCreate(&cudnnHandle);
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("Cannot create cuDNN handle, ") + cudnnGetErrorString(status));
    status = cudnnSetStream(cudnnHandle, cudaStream);
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(cudnnGetErrorString(status));

    // create cuBLAS handle
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Cannot create cuBLAS handle.");
    if (cublasSetStream(cublasHandle, cudaStream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Cannot set CUDA stream for cuBLAS.");

    // attach the allocator thread to the current CUDA device
    int currentDevice;
    if (cudaGetDevice(&currentDevice) != cudaError::cudaSuccess)
        throw std::runtime_error("cudaGetDevice failed");
    allocator.call(this, &CUDA::attachDevice, currentDevice);

    // Query GPU properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, currentDevice);
    cudnn::Context::raiseIfError("cudaGetDeviceProperties failed");
    registersPerThreadBlock = deviceProp.regsPerBlock;
    alignmentConstraint = deviceProp.textureAlignment;
}


void* CUDA::malloc(size_t size) {
    void* memory;
    allocator.call(this, &CUDA::internalMalloc, size, memory);
    return memory;
}


void CUDA::free(void* memory) {
    allocator.call(this, &CUDA::internalFree, memory);
}

}
}