#include <cuda.h>

#include <stdexcept>

#include "context.hpp"
#include "kernels.hpp"
#include "hidenames.h"

using namespace upstride;

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
__global__ void HIDENAME(cropNCHW)(const T* in, T* out,
                                   unsigned int dx, unsigned dy,
                                   unsigned int inWidth, unsigned int inHeight,
                                   unsigned int outWidth, unsigned int outHeight,
                                   unsigned int depth) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < outWidth && y < outHeight && z < depth)
        out[(z * outHeight + y) * outWidth + x] = in[(z * inHeight + y + dy) * inWidth + x + dx];
}

/**
 * @brief CUDA kernel cropping an input NHWC tensor along H and W dimensions
 * C axis is tiled with X grid axis.
 * NHW subspace is tiled with Y grid axis.
 * @param in            pointer to input values
 * @param out           pointer to output values
 * @param dx            horizontal shift
 * @param dy            vertical shift
 * @param inWidth       input tensor width
 * @param inHeight      input tensor height
 * @param outWidth      output tensor width
 * @param outHeight     output tensor height
 * @param channels      number of channels
 * @param batchSize     number of images in batch
 */
 template <typename T>
 __global__ void HIDENAME(cropNHWC)(const T* in, T* out,
                                    unsigned int dx, unsigned dy,
                                    unsigned int inWidth, unsigned int inHeight,
                                    unsigned int outWidth, unsigned int outHeight,
                                    unsigned int channels, unsigned int batchSize) {
    // get channel number and position in NHW subspace
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = blockIdx.y * blockDim.y + threadIdx.y;
    // get in-batch image number and pixel position in HW plane
    unsigned int n = p / (outWidth * outHeight);
    p -= n * outWidth * outHeight;
    unsigned int y = p / outWidth, x = p - y * outWidth;
    // write input value to output if in bounds
    if (n < batchSize && y < outHeight && c < channels)
        out[((n * outHeight + y) * outWidth + x) * channels + c] = in[((n * inHeight + y + dy) * inWidth + x + dx) * channels + c];
}

/**
 * @brief CUDA kernel inserting an input NCHW tensor into an output NCHW tensor
 * @param in            pointer to input values
 * @param out           pointer to output values
 * @param dx            horizontal shift
 * @param dy            vertical shift
 * @param inWidth       input tensor width
 * @param inHeight      input tensor height
 * @param outWidth      output tensor width
 * @param outHeight     output tensor height
 * @param depth         the depth of both input and output tensors (N times C)
 */
template <typename T>
__global__ void HIDENAME(insertNCHW)(const T* in, T* out,
                                     unsigned int dx, unsigned int dy,
                                     unsigned int inWidth, unsigned int inHeight,
                                     unsigned int outWidth, unsigned int outHeight,
                                     unsigned int depth) {
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z >= depth)
        return;
    unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int oy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int o = (z * outHeight + oy) * outWidth + ox;
    unsigned int ix = ox - dx;
    unsigned int iy = oy - dy;
    if (dx <= ox && dy <= oy && ix < inWidth && iy < inHeight)
        out[o] = in[(z * inHeight + iy) * inWidth + ix];
    else if (ox < outWidth && oy < outHeight)
        out[o] = 0;
}

/**
 * @brief CUDA kernel inserting an input NHWC tensor into an output NHWC tensor
 * C axis is tiled with X grid axis.
 * NHW subspace is tiled with Y grid axis.
 * @param in            pointer to input values
 * @param out           pointer to output values
 * @param dx            horizontal shift
 * @param dy            vertical shift
 * @param inWidth       input tensor width
 * @param inHeight      input tensor height
 * @param outWidth      output tensor width
 * @param outHeight     output tensor height
 * @param channels      number of channels
 * @param batchSize     number of images in batch
 */
template <typename T>
__global__ void HIDENAME(insertNHWC)(const T* in, T* out,
                                     unsigned int dx, unsigned int dy,
                                     unsigned int inWidth, unsigned int inHeight,
                                     unsigned int outWidth, unsigned int outHeight,
                                     unsigned int channels, unsigned int batchSize) {
    // get channel number and position in NHW subspace
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = blockIdx.y * blockDim.y + threadIdx.y;
    // get in-batch image number and pixel position in output HW plane
    unsigned int n = p / (outWidth * outHeight);
    p -= n * outWidth * outHeight;
    unsigned int oy = p / outWidth, ox = p - oy * outWidth;
    if (n >= batchSize && c >= channels)
        return;
    // get position in input HW plane
    unsigned int ix = ox - dx;
    unsigned int iy = oy - dy;
    // compute linear index in output
    uint64_t o = ((n * outHeight + oy) * outWidth + ox) * channels + c;
    // copy input value if in bounds, zero otherwise
    if (dx <= ox && dy <= oy && ix < inWidth && iy < inHeight)
        out[o] = in[((n * inHeight + iy) * inWidth + ix) * channels + c];
    else
        out[o] = 0;
}

template <typename T>
__global__ void HIDENAME(addBiasNCHW)(T* tensor, const T* bias, unsigned int tensorSize, unsigned int imageSize, unsigned int channels) {
    unsigned int pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int channel = (pos / imageSize) % channels;
    if (pos < tensorSize)
        tensor[pos] += bias[channel];
}

template <typename T>
__global__ void HIDENAME(addBiasNHWC)(T* tensor, const T* bias, unsigned int tensorSize, unsigned int channels) {
    unsigned int pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int channel = pos % channels;
    if (pos < tensorSize)
        tensor[pos] += bias[channel];
}

template <typename T>
__global__ void HIDENAME(addBiasNC)(T* tensor, const T* bias, unsigned int tensorSize, unsigned int channels) {
    unsigned int pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int channel = pos % channels;
    if (pos < tensorSize)
        tensor[pos] += bias[channel];
}


/**
 * @brief Sets up a simple CUDA kernel grid config for a pointwise operation.
 * For channels-first (NCHW) tensor format this routine generates a weird thread repartition to be revisited
 * For channels-last (NHWC) tensor format, the produced configuration is as follows:
 *  - C axis is tiled with X grid axis.
 *  - NHW subspace is tiled with Y grid axis.
 * @param shape         shape of the thread space to sample
 * @param dataFormat    data format of the corresponding shape
 * @param threads       number of threads per block (output)
 * @param blocks        number of thread blocks (ouptut)
 * @param device        CUDA device
 */
inline static void makeGridConfig(const Shape& shape, DataFormat dataFormat, dim3& threads, dim3& blocks, const device::CUDA& device) {
    if (dataFormat == DataFormat::NCHW) {
        const int64_t depth = shape.depth(dataFormat) * shape[0];
        const int64_t z = std::min<int64_t>(device.getMaxBlockSize().z, depth);
        const int64_t xy = (int64_t)std::sqrt(device.getMaxThreadsPerBlock() / z);
        threads = dim3(xy, xy, z);
        blocks = dim3(
            ceili(shape.width(dataFormat), threads.x),
            ceili(shape.height(dataFormat), threads.y),
            ceili(depth, threads.z));
    }
    else if (dataFormat == DataFormat::NHWC) {
        const int64_t xThreads = std::min<int64_t>(device.getMaxThreadsPerBlock(), shape.depth(dataFormat));
        const int64_t nhwSize = shape[0] * shape.height(dataFormat) * shape.width(dataFormat);
        threads = dim3(
            xThreads,
            std::min(device.getMaxThreadsPerBlock() / xThreads, nhwSize)
        );
        blocks = dim3(
            ceili(shape.depth(dataFormat), threads.x),
            ceili(nhwSize, threads.y));
    }
    else
        throw std::runtime_error(std::string("Unsupported data format: ") + dataFormatToString(dataFormat));
}


template <typename T>
void crop(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset) {
    const Shape& inShape = input.getShape();
    const Shape& outShape = output.getShape();

    if (inShape.getSize() != 4 || outShape.getSize() != 4)
        throw std::runtime_error("Expecting four-dimenisonal input and output tensors");
    if (outShape.width(dataFormat) + offset.y > inShape.width(dataFormat) ||
        outShape.height(dataFormat) + offset.x > inShape.height(dataFormat))
        throw std::runtime_error("Cannot fit output tensor into input tensor");
    if (inShape.depth(dataFormat) != outShape.depth(dataFormat))
        throw std::runtime_error("Input / output depth mismatch");
    if (inShape[0] != outShape[0])
        throw std::runtime_error("Input / output batch size mismatch");

    // setting up thread grid
    dim3 threads, blocks;
    makeGridConfig(inShape, dataFormat, threads, blocks, output.getDevice());

    // launching the kernel
    if (dataFormat == DataFormat::NCHW)
        HIDENAME(cropNCHW)<<<blocks, threads, 0, input.getDevice().stream()>>>(
            input.getDataPtr(),
            output.getDataPtr(),
            offset.y, offset.x,
            inShape.width(dataFormat), inShape.height(dataFormat),
            outShape.width(dataFormat), outShape.height(dataFormat),
            inShape.depth(dataFormat) * inShape[0]);
    else if (dataFormat == DataFormat::NHWC)
        HIDENAME(cropNHWC)<<<blocks, threads, 0, input.getDevice().stream()>>>(
            input.getDataPtr(),
            output.getDataPtr(),
            offset.y, offset.x,
            inShape.width(dataFormat), inShape.height(dataFormat),
            outShape.width(dataFormat), outShape.height(dataFormat),
            inShape.depth(dataFormat), inShape[0]);

    cudnn::Context::raiseIfError();
}

template<typename T>
void addBias(Tensor<device::CUDA, T>& tensor, const Tensor<device::CUDA, const T>& bias, DataFormat dataFormat) {
    const Shape& shape = tensor.getShape();
    unsigned int tensorSize = shape.numel();

    dim3 threads{tensor.getDevice().getMaxThreadsPerBlock(), 1, 1};
    dim3 blocks = dim3(ceili(tensorSize, threads.x), 1, 1);

    if (dataFormat == DataFormat::NCHW) {
        if (shape.getSize() != 4)
            throw std::runtime_error("Expecting a four-dimensional tensor");

        unsigned int channels = shape[1];
        unsigned int height = shape.height(dataFormat);
        unsigned int width = shape.width(dataFormat);
        unsigned int imageSize = height * width;

        HIDENAME(addBiasNCHW)<<<blocks, threads, 0, tensor.getDevice().stream()>>>(
            tensor.getDataPtr(), bias.getDataPtr(), tensorSize, imageSize, channels
        );
    }
    else if (dataFormat == DataFormat::NHWC) {
        if (shape.getSize() != 4)
            throw std::runtime_error("Expecting a four-dimensional tensor");

        unsigned int channels = shape[3];
        if (channels != bias.getShape().numel())
            throw std::runtime_error("Tensor and bias sizes mismatch");

        HIDENAME(addBiasNHWC)<<<blocks, threads, 0, tensor.getDevice().stream()>>>(
            tensor.getDataPtr(), bias.getDataPtr(), tensorSize, channels
        );
    }
    else if (dataFormat == DataFormat::NC) {
        if (shape.getSize() != 2)
            throw std::runtime_error("Expecting a two-dimensional tensor");

        unsigned int channels = shape[1];
        if (channels != bias.getShape().numel())
            throw std::runtime_error("Tensor and bias sizes mismatch");

        HIDENAME(addBiasNC)<<<blocks, threads, 0, tensor.getDevice().stream()>>>(
            tensor.getDataPtr(), bias.getDataPtr(), tensorSize, channels
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
    makeGridConfig(outShape, dataFormat, threads, blocks, output.getDevice());

    // launching the kernel
    if (dataFormat == DataFormat::NCHW)
        HIDENAME(insertNCHW)<<<blocks, threads, 0, input.getDevice().stream()>>>(
            input.getDataPtr(),
            output.getDataPtr(),
            offset.y, offset.x,
            inShape.width(dataFormat), inShape.height(dataFormat),
            outShape.width(dataFormat), outShape.height(dataFormat),
            outShape.depth(dataFormat) * outShape[0]);
    else if (dataFormat == DataFormat::NHWC)
        HIDENAME(insertNHWC)<<<blocks, threads, 0, input.getDevice().stream()>>>(
            input.getDataPtr(),
            output.getDataPtr(),
            offset.y, offset.x,
            inShape.width(dataFormat), inShape.height(dataFormat),
            outShape.width(dataFormat), outShape.height(dataFormat),
            inShape.depth(dataFormat), inShape[0]);

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
    const int numThreads = device.getMaxThreadsPerBlock();
    ::HIDENAME(accumulateAdd)<<<ceili(length, numThreads), numThreads, 0, device.stream()>>>(accumulator, term, length);
}

template <>
void accumulateSub(const device::CUDA& device, float* accumulator, const float* term, int length) {
    const int numThreads = device.getMaxThreadsPerBlock();
    ::HIDENAME(accumulateSub)<<<ceili(length, numThreads), numThreads, 0, device.stream()>>>(accumulator, term, length);
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
    const int numThreads = device.getMaxThreadsPerBlock();
    ::HIDENAME(accumulateAdd)<<<ceili(length, numThreads), numThreads, 0, device.stream()>>>(accumulator, term, length);
    cudnn::Context::raiseIfError();
}

template <>
void accumulateSub(const device::CUDA& device, half* accumulator, const half* term, int length) {
    const int numThreads = device.getMaxThreadsPerBlock();
    ::HIDENAME(accumulateSub)<<<ceili(length, numThreads), numThreads, 0, device.stream()>>>(accumulator, term, length);
    cudnn::Context::raiseIfError();
}
#endif

}  // namespace cudnn

}  // namespace upstride