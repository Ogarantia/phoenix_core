#include <cuda.h>

#include <stdexcept>

#include "context.hpp"
#include "kernels.hpp"
#include "hidenames.h"

using namespace upstride;

static const int NUM_THREADS = 1024;  //!< default number of CUDA threads per block

template <typename T>
__global__ void HIDENAME(accumulateAdd)(T* acc, const T* term, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        acc[i] += term[i];
}

template <typename T>
__global__ void HIDENAME(accumulateSub)(T* acc, const T* term, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        acc[i] -= term[i];
}

/**
 * @brief CUDA kernel cropping an input NCHW tensor along H and W dimensions
 * The output tensor is smaller or equal in size than the input tensor.
 * @param in            pointer to input values
 * @param out           pointer to output values
 * @param dx            horizontal shift
 * @param dy            vertical shift
 * @param inWidth       input tensor width
 * @param inHeight      input tensor height
 * @param outWidth      output tensor width
 * @param outHeight     output tensor height
 * @param depth         the depth of both input and output tensors (N times C times the element size)
 */
template <typename T>
__global__ void HIDENAME(cropNCHW)(const T* in, T* out, int dx, int dy, int inWidth, int inHeight, int outWidth, int outHeight, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < outWidth && y < outHeight && z < depth)
        out[(z * outHeight + y) * outWidth + x] = in[(z * inHeight + y + dy) * inWidth + x + dx];
}

/**
 * @brief CUDA kernel inserting an input NCHW tensor into an output NCHW tensor
 * The input tensor is smaller or equal in size than the output tensor.
 * @param in            pointer to input values
 * @param out           pointer to output values
 * @param dx            horizontal shift
 * @param dy            vertical shift
 * @param inWidth       input tensor width
 * @param inHeight      input tensor height
 * @param outWidth      output tensor width
 * @param outHeight     output tensor height
 * @param depth         the depth of both input and output tensors (N times C times the element size)
 */
template <typename T>
__global__ void HIDENAME(insertNCHW)(const T* in, T* out, int dx, int dy, int inWidth, int inHeight, int outWidth, int outHeight, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < inWidth && y < inHeight && z < depth)
        out[(z * outHeight + y + dy) * outWidth + x + dx] = in[(z * inHeight + y) * inWidth + x];
}

template <typename T>
__global__ void HIDENAME(addBiasNCHW)(T* tensor, const T* bias, int tensorSize, int imageSize, int depth) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int channel = (pos / imageSize) % depth;
    if (pos < tensorSize)
        tensor[pos] += bias[channel];
}

template <typename T>
__global__ void HIDENAME(addBiasNC)(T* tensor, const T* bias, int tensorSize, int depth) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int channel = pos % depth;
    if (pos < tensorSize)
        tensor[pos] += bias[channel];
}


/**
 * @brief Sets up a simple CUDA kernel grid config for a pointwise operation
 * @param shape         shape of the threads space to sample
 * @param dataFormat    data format of the corresponding shape
 * @param threads       number of threads per block (output)
 * @param blocks        number of thread blocks (ouptut)
 * @param numThreads    maximum number of threads per block
 */
inline static void makeGridConfig(const Shape& shape, DataFormat dataFormat, dim3& threads, dim3& blocks, const int numThreads = NUM_THREADS) {
    const int depth = shape.depth(dataFormat) * shape[0];
    const int z = std::min(cudnn::Context::MAX_BLOCK_DEPTH, depth);
    const int xy = (int)std::sqrt(numThreads / z);
    threads = dim3(xy, xy, z);
    blocks = dim3(
        ceili(shape.width(dataFormat), threads.x),
        ceili(shape.height(dataFormat), threads.y),
        ceili(depth, threads.z));
}

template <typename T>
void crop(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset) {
    // check stuff
    const Shape& inShape = input.getShape();
    const Shape& outShape = output.getShape();

    if (dataFormat != DataFormat::NCHW)
        throw std::runtime_error("Unsupported data format");
    if (inShape.getSize() != 4 || outShape.getSize() != 4)
        throw std::runtime_error("Expecting four-dimenisonal input and output tensors");
    if (outShape.width(dataFormat) + offset.y < inShape.width(dataFormat) ||
        outShape.height(dataFormat) + offset.x < inShape.height(dataFormat))
        throw std::runtime_error("Cannot fit output tensor into input tensor");
    if (inShape.depth(dataFormat) != outShape.depth(dataFormat))
        throw std::runtime_error("Input / output depth mismatch");
    if (inShape[0] != outShape[0])
        throw std::runtime_error("Input / output batch size mismatch");

    // setting up thread grid
    dim3 threads, blocks;
    makeGridConfig(inShape, dataFormat, threads, blocks);

    // launching the kernel
    HIDENAME(cropNCHW)<<<blocks, threads, 0, input.getDevice().stream()>>>(
        input.getDataPtr(),
        output.getDataPtr(),
        offset.y, offset.x,
        inShape.width(dataFormat), inShape.height(dataFormat),
        outShape.width(dataFormat), outShape.height(dataFormat),
        inShape.depth(dataFormat) * inShape[0]);

    cudnn::Context::raiseIfError();
}

template<typename T>
void addBias(Tensor<device::CUDA, T>& tensor, const Tensor<device::CUDA, const T>& bias, DataFormat dataFormat) {
    if (dataFormat != DataFormat::NCHW && dataFormat != DataFormat::IO)
        throw std::runtime_error("Unsupported data format");
    const Shape& shape = tensor.getShape();
    if (dataFormat == DataFormat::NCHW && shape.getSize() != 4)
        throw std::runtime_error("Expecting a four-dimensional tensor");
    if (dataFormat == DataFormat::IO && shape.getSize() != 2)
        throw std::runtime_error("Expecting a two-dimensional tensor");
    if (shape.depth(dataFormat) != bias.getShape().numel())
        throw std::runtime_error("Tensor and bias sizes mismatch");

    unsigned int depth = shape.depth(dataFormat);
    unsigned int tensorSize = shape.numel();

    dim3 threads{NUM_THREADS, 1, 1};
    dim3 blocks = dim3(ceili(tensorSize, threads.x), 1, 1);

    if (dataFormat == DataFormat::NCHW) {
        unsigned int height = shape.height(dataFormat);
        unsigned int width = shape.width(dataFormat);
        unsigned int imageSize = height * width;

        HIDENAME(addBiasNCHW)<<<blocks, threads, 0, tensor.getDevice().stream()>>>(
            tensor.getDataPtr(), bias.getDataPtr(), tensorSize, imageSize, depth
        );
    }
    else if (dataFormat == DataFormat::IO) {
        HIDENAME(addBiasNC)<<<blocks, threads, 0, tensor.getDevice().stream()>>>(
            tensor.getDataPtr(), bias.getDataPtr(), tensorSize, depth
        );
    }
    else
        throw std::runtime_error("addBias is currently not implemented for the given dataFormat.");
}

template <typename T>
void insert(const Tensor<device::CUDA, const T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset) {
    // check stuff
    const Shape& inShape = input.getShape();
    const Shape& outShape = output.getShape();

    if (dataFormat != DataFormat::NCHW)
        throw std::runtime_error("Unsupported data format");
    if (inShape.getSize() != 4 || outShape.getSize() != 4)
        throw std::runtime_error("Expecting four-dimenisonal input and output tensors");
    if (inShape.width(dataFormat) + offset.y > outShape.width(dataFormat) ||
        inShape.height(dataFormat) + offset.x > outShape.height(dataFormat))
        throw std::runtime_error("Cannot fit input tensor into output tensor");
    if (inShape.depth(dataFormat) != outShape.depth(dataFormat))
        throw std::runtime_error("Input / output depth mismatch");
    if (inShape[0] != outShape[0])
        throw std::runtime_error("Input / output batch size mismatch");

    // setting up thread grid
    dim3 threads, blocks;
    makeGridConfig(outShape, dataFormat, threads, blocks);

    // launching the kernel
    HIDENAME(insertNCHW)<<<blocks, threads, 0, input.getDevice().stream()>>>(
        input.getDataPtr(),
        output.getDataPtr(),
        offset.y, offset.x,
        inShape.width(dataFormat), inShape.height(dataFormat),
        outShape.width(dataFormat), outShape.height(dataFormat),
        outShape.depth(dataFormat) * outShape[0]);

    cudnn::Context::raiseIfError();
}

namespace upstride {
namespace cudnn {

template <>
void crop(const Tensor<device::CUDA, float>& input, Tensor<device::CUDA, float>& output, DataFormat dataFormat, const IntPair& offset) {
    ::crop(input, output, dataFormat, offset);
}

template <>
void insert(const Tensor<device::CUDA, const float>& input, Tensor<device::CUDA, float>& output, DataFormat dataFormat, const IntPair& offset) {
    ::insert(input, output, dataFormat, offset);
}

template<>
void addBias(Tensor<device::CUDA, float>& tensor, const Tensor<device::CUDA, const float>& bias, DataFormat dataFormat) {
    ::addBias(tensor, bias, dataFormat);
}

template <>
void accumulateAdd(const device::CUDA& device, float* accumulator, const float* term, int length) {
    ::HIDENAME(accumulateAdd)<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, device.stream()>>>(accumulator, term, length);
}

template <>
void accumulateSub(const device::CUDA& device, float* accumulator, const float* term, int length) {
    ::HIDENAME(accumulateSub)<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, device.stream()>>>(accumulator, term, length);
}

#ifdef UPSTRIDE_ENABLE_FP16
template <>
void crop(const Tensor<device::CUDA, half>& input, Tensor<device::CUDA, half>& output, DataFormat dataFormat, const IntPair& offset) {
    ::crop(input, output, dataFormat, offset);
}

template <>
void insert(const Tensor<device::CUDA, const half>& input, Tensor<device::CUDA, half>& output, DataFormat dataFormat, const IntPair& offset) {
    ::insert(input, output, dataFormat, offset);
}

template<>
void addBias(Tensor<device::CUDA, half>& tensor, const Tensor<device::CUDA, const half>& bias, DataFormat dataFormat) {
    ::addBias(tensor, bias, dataFormat);
}

template <>
void accumulateAdd(const device::CUDA& device, half* accumulator, const half* term, int length) {
    ::HIDENAME(accumulateAdd)<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, device.stream()>>>(accumulator, term, length);
    cudnn::Context::raiseIfError();
}

template <>
void accumulateSub(const device::CUDA& device, half* accumulator, const half* term, int length) {
    ::HIDENAME(accumulateSub)<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, device.stream()>>>(accumulator, term, length);
    cudnn::Context::raiseIfError();
}
#endif

}  // namespace cudnn

}  // namespace upstride