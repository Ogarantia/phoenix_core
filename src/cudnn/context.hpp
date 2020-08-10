/**
 * @file context.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief cuDNN context shared among all the operations executed with cuDNN
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <cudnn.h>

#include <exception>

#include "../upstride.hpp"

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

/**
 * @brief cuDNN-specific shareable singleton context
 */
class Context : public upstride::Context {
   private:
    cudnnHandle_t handle;

    Context(const int typeDim) : upstride::Context(typeDim) {
        raiseIfError(cudnnCreate(&handle));
    }

    ~Context() {
        raiseIfError(cudnnDestroy(handle));
    }

   public:
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
     * @brief Provides the instance of cuDNN context.
     * @return the context
     */
    static Context& getInstance();

    const cudnnHandle_t& getHandle() const { return handle; }
};

/**
 * @brief Pointer to a device memory and its allocation/disposition routines
 * Often acts as a pointer due to conversion operators.
 */
class Memory {
   private:
    void* ptr;
    size_t size;

    Memory(const Memory&) = delete;  // deleting copying constructor
   public:
    Memory() : ptr(nullptr), size(0) {}
    Memory(size_t sizeBytes);
    Memory(Memory&&);
    Memory& operator=(Memory&&);
    ~Memory();

    /**
     * @brief Frees the pointed memory if any.
     */
    void free();

    /**
     * @brief Checks pointer validity.
     *
     * @return true if the pointer points to a valid memory.
     * @return false otherwise
     */
    inline operator bool() const { return ptr != nullptr; }

    template <typename T>
    inline operator T*() { return static_cast<T*>(ptr); }

    template <typename T>
    inline operator const T*() const { return static_cast<const T*>(ptr); }
};

/**
 * @brief Fills a cuDNN tensor descriptor
 * 
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
