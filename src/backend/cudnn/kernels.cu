#include <cuda.h>

#include <stdexcept>

#include "context.hpp"
#include "kernels.hpp"

using namespace upstride;

static const int NUM_THREADS = 64;  //!< default number of threads

template <typename T>
__global__ void accumulateAdd(T* acc, const T* term, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
        acc[i] += term[i];
}

template <typename T>
__global__ void accumulateSub(T* acc, const T* term, int length) {
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
__global__ void cropNCHW(const T* in, T* out, int dx, int dy, int inWidth, int inHeight, int outWidth, int outHeight, int depth) {
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
__global__ void insertNCHW(const T* in, T* out, int dx, int dy, int inWidth, int inHeight, int outWidth, int outHeight, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < inWidth && y < inHeight && z < depth)
        out[(z * outHeight + y + dy) * outWidth + x + dx] = in[(z * inHeight + y) * inWidth + x];
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
    const int z = std::min(numThreads, depth);
    const int xy = (int)std::sqrt(numThreads / z);
    threads = dim3(xy, xy, z);
    blocks = dim3(
        ceili(shape.width(dataFormat), threads.x),
        ceili(shape.height(dataFormat), threads.y),
        ceili(shape.depth(dataFormat) * shape[0], threads.z));
}

namespace upstride {
namespace cudnn {

template <>
void crop(const Tensor<device::CUDA, float>& input, Tensor<device::CUDA, float>& output, DataFormat dataFormat, const IntPair& offset) {
    // check stuff
    const Shape& inShape = input.getShape();
    const Shape& outShape = output.getShape();

    if (dataFormat != DataFormat::NCHW)
        throw std::runtime_error("Unsupported data format");
    if (inShape.getSize() != 4 || outShape.getSize() != 4)
        throw std::runtime_error("Expecting four-dimenisonal input and output tensors");
    if (outShape.width(dataFormat) + offset.x < inShape.width(dataFormat) ||
        outShape.height(dataFormat) + offset.y < inShape.height(dataFormat))
        throw std::runtime_error("Cannot fit output tensor into input tensor");
    if (inShape.depth(dataFormat) != outShape.depth(dataFormat))
        throw std::runtime_error("Input / output depth mismatch");
    if (inShape[0] != outShape[0])
        throw std::runtime_error("Input / output batch size mismatch");

    // setting up thread grid
    dim3 threads, blocks;
    makeGridConfig(inShape, dataFormat, threads, blocks);

    // launching the kernel
    cropNCHW<<<blocks, threads>>>(
        input.getDataPtr(),
        output.getDataPtr(),
        offset.x, offset.y,
        inShape.width(dataFormat), inShape.height(dataFormat),
        outShape.width(dataFormat), outShape.height(dataFormat),
        inShape.depth(dataFormat) * inShape[0]);

    Context::raiseIfError();
}

template <>
void insert(const Tensor<device::CUDA, const float>& input, Tensor<device::CUDA, float>& output, DataFormat dataFormat, const IntPair& offset) {
    // check stuff
    const Shape& inShape = input.getShape();
    const Shape& outShape = output.getShape();

    if (dataFormat != DataFormat::NCHW)
        throw std::runtime_error("Unsupported data format");
    if (inShape.getSize() != 4 || outShape.getSize() != 4)
        throw std::runtime_error("Expecting four-dimenisonal input and output tensors");
    if (inShape.width(dataFormat) + offset.x > outShape.width(dataFormat) ||
        inShape.height(dataFormat) + offset.y > outShape.height(dataFormat))
        throw std::runtime_error("Cannot fit input tensor into output tensor");
    if (inShape.depth(dataFormat) != outShape.depth(dataFormat))
        throw std::runtime_error("Input / output depth mismatch");
    if (inShape[0] != outShape[0])
        throw std::runtime_error("Input / output batch size mismatch");

    // setting up thread grid
    dim3 threads, blocks;
    makeGridConfig(outShape, dataFormat, threads, blocks);

    // launching the kernel
    insertNCHW<<<blocks, threads>>>(
        input.getDataPtr(),
        output.getDataPtr(),
        offset.x, offset.y,
        inShape.width(dataFormat), inShape.height(dataFormat),
        outShape.width(dataFormat), outShape.height(dataFormat),
        outShape.depth(dataFormat) * outShape[0]);

    Context::raiseIfError();
}

template <>
void accumulateAdd(float* accumulator, const float* term, int length) {
    ::accumulateAdd<<<ceili(length, NUM_THREADS), NUM_THREADS>>>(accumulator, term, length);
}

template <>
void accumulateSub(float* accumulator, const float* term, int length) {
    ::accumulateSub<<<ceili(length, NUM_THREADS), NUM_THREADS>>>(accumulator, term, length);
}

}  // namespace cudnn

}  // namespace upstride