#pragma once
#include <cuda.h>
#include <cudnn.h>

namespace upstride {
namespace device {
class CUDA {
   private:
    cudaStream_t cudaStream;
    cudnnHandle_t cudnnHandle;

    CUDA(const CUDA&) = delete;  // disable copying

   public:
    inline CUDA(const cudaStream_t& stream) : cudaStream(stream) {
        auto status = cudnnCreate(&cudnnHandle);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(cudnnGetErrorString(status));
        status = cudnnSetStream(cudnnHandle, cudaStream);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(cudnnGetErrorString(status));
    }

    inline ~CUDA() {
        cudnnDestroy(cudnnHandle);
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
};
}  // namespace device
}  // namespace upstride