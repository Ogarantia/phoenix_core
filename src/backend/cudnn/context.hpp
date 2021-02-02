/**
 * @file context.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief cuDNN context shared among all the operations executed with cuDNN
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <map>
#include <stdexcept>
#include <mutex>

#include "../backend.hpp"
#include "device.hpp"
#include "half.hpp"

namespace upstride {
namespace cudnn {

/**
 * @brief Retrieves cuDNN tensor format corresponding to a specific dataFormat by UpStride
 * @param dataFormat
 * @return cudnnTensorFormat_t
 */
static inline cudnnTensorFormat_t dataFormatToTensorFormat(DataFormat dataFormat) {
    switch (dataFormat) {
        case DataFormat::NCHW:
            return CUDNN_TENSOR_NCHW;
        case DataFormat::NHWC:
            return CUDNN_TENSOR_NHWC;
        default:
            throw std::invalid_argument("Unimplemented DataFormat.");
    }
}

/**
 * @brief Retrieves cuDNN datatype handle corresponding to a common POD data type
 * @tparam T    The input type
 * @return cudnnDataType_t
 */
template <typename T>
static inline cudnnDataType_t getDataType();

template <>
inline cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT; }

template <>
inline cudnnDataType_t getDataType<half>() { return CUDNN_DATA_HALF; }

/**
 * @brief cuDNN-specific shareable singleton context
 */
class Context : public upstride::Context {
   private:
    std::map<cudaStream_t, device::CUDA> devices;  //!< the devices; they are indexed by CUDA streams
    std::mutex mutex;

   protected:
    void cleanUp();

   public:
    static const int MAX_BLOCK_DEPTH;      //!< maximum number of CUDA threads per block along Z dimension

    Context() {}
    ~Context() {}

    /**
     * @brief Checks a cuDNN operation status and throws an exception if the operation was not successful with the a message containing the status.
     * @param status    The status value to check
     */
    inline static void raiseIfError(const cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(cudnnGetErrorString(status));
    }

    /**
     * @brief Checks a CUDA operation error code and throws an exception if the operation was not successful.
     * @param status    The status value to check
     */
    inline static void raiseIfError(const cudaError_t error) {
        if (error != cudaError::cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    /**
     * @brief Checks the last CUDA operation error code and throws an exception if the operation was not successful.
     */
    inline static void raiseIfError() {
        auto error = cudaPeekAtLastError();
        if (error != cudaError::cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    /**
     * @brief Checks the last CUDA operation error code, if the operation was not successful prints message and throws an exception.
     * @param msg     Additional message to be printed before throwing the error
     */
    inline static void raiseIfError(const char* msg) {
        auto error = cudaPeekAtLastError();
        if (error != cudaError::cudaSuccess) {
            std::cerr << msg << "\n";
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    /**
     * @brief Checks a cuBLAS operation status and throws an exception if the operation was not successful with the a message containing the status.
     * @param status    The status value to check
     */
    inline static void raiseIfError(const cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(cublasGetErrorString(status));
    }

    /**
     * @brief Returns the message associated to a given cuBLAS status.
     * @param status    The status value to be translated into plain text
     */
    inline static const char* cublasGetErrorString(const cublasStatus_t status)
    {
        switch(status)
        {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
            default : return "unknown error";
        }
    }

    /**
     * @brief Retrieves or creates a device attached to a specific CUDA stream.
     * @param stream the CUDA stream
     * @return the device instance.
     */
    device::CUDA& registerDevice(const cudaStream_t& stream);
};


/**
 * @brief Fills a cuDNN tensor descriptor
 * @tparam T scalar datatype
 * @param descriptor    the descriptor to fill
 * @param shape         tensor shape
 * @param dataFormat    tensor data format
 */
template <typename T>
static inline void setTensorDescriptor(cudnnTensorDescriptor_t& descriptor, const Shape& shape, DataFormat dataFormat) {
    cudnn::Context::raiseIfError(cudnnSetTensor4dDescriptor(
        descriptor,
        cudnn::dataFormatToTensorFormat(dataFormat), cudnn::getDataType<T>(),
        shape[0], shape.depth(dataFormat), shape.height(dataFormat), shape.width(dataFormat)));
}

}  // namespace cudnn
}  // namespace upstride
