#include <cuda.h>

#include <exception>

#include "context.hpp"
#include "kernels.hpp"

using namespace upstride;

/**
 * @brief CUDA kernel cropping an input NCHW tensor along H and W dimensions
 * 
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
 * @brief Rounding up integer division
 * 
 * @param n nominator
 * @param d denominator
 * @return closest integer greater than n/d 
 */
inline int ceili(int n, int d) {
    return (n + d - 1) / d;
}

void upstride::cudnn::crop(const Tensor<const float>& input, Tensor<float>& output, DataFormat dataFormat, const IntPair& offset) {
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
    static const int NUM_THREADS = 64;
    const int depth = outShape.depth(dataFormat) * outShape[0];
    const int z = std::min(NUM_THREADS, depth);
    const int xy = (int)std::sqrt(NUM_THREADS / z);
    const dim3 threads(xy, xy, z);
    const dim3 blocks(
        ceili(outShape.width(dataFormat), threads.x),
        ceili(outShape.height(dataFormat), threads.y),
        ceili(outShape.depth(dataFormat) * outShape[0], threads.z));

    // launching the kernel
    cropNCHW<<<blocks, threads>>>(
        input.getDataPtr(),
        output.getDataPtr(),
        offset.x, offset.y,
        inShape.width(dataFormat), inShape.height(dataFormat),
        outShape.width(dataFormat), outShape.height(dataFormat),
        depth);

    Context::raiseIfError();
}