#include <type_traits>
#include "quat_pointwise_conv2d.hpp"

namespace upstride {
namespace cuda {

template<typename T>
struct quat { T r, i, j, k; };


/**
 * @brief Forward pass convolution CUDA kernel
 *
 * This kernel works properly only for blockDimX == blockDimY being a power of 2.
 * blockDim.z should be equal to the batch size.
 *
 * With this kernel, each thread computes one output value.
 * Each thread block corresponds to a (blockDimX x blockDimY) patch of the output values.
 *
 * A thread block shares memory, in each input processing step each thread fetches one input and one weights value,
 * and then calculates the contribution of blockDimX (== blockDimY) factors (convolution multiplications and additions)
 * to the final output value.
 *
 * After each input processing step the window of values to fetch moves across the input channels dimension
 * for input fetching and across the output channels dimension for weights fetching.
 *
 * Intuitively (disregarding batchSize and channel first):
 * Input dimensions: x - imageSize, y - inputChannels
 * Weights dimensions: x - inputChannels, y - outputChannels
 * Output dimensions: x - imageSize, y - outputChannels
 */
template<typename T, unsigned int blockDimX, unsigned int blockDimY, unsigned int bankSkew>
__global__ void pointwise_conv_forward_2D_shared_mem(
    const T* input, const T* weights, const T* bias, T* output,
    const int batchSize, const int inputChannels, const int outputChannels, const int imageSize
) {
    static_assert(blockDimX == blockDimY, "pointwise_conv_forward_2D_shared_mem kernel requires equal x and y thread block dimensions");
    static_assert((blockDimY & (blockDimY - 1)) == 0, "pointwise_conv_forward_2D_shared_mem requires y thread block dimension to be a power of 2");

    // declare CUDA shared memory
    // use banks skew to avoid bank conflicts in shared memory (seemingly not much of a difference though)
    __shared__ quat<T> inputBuffer[blockDimY * blockDimX];
    __shared__ quat<T> weightsBuffer[(blockDimX + bankSkew) * blockDimY];

    // determine the position of the output value this thread corresponds to
    const int outChannel = blockIdx.y * blockDimY + threadIdx.y;                // output channel
    const int pix = blockIdx.x * blockDimX + threadIdx.x;                       // pixel number in image
    const int batchPos = blockIdx.z;                                            // image number in batch

    // use an accumulator to store the intermediate partial convolution values
    quat<T> acc{ 0, 0, 0, 0 };

    // used to unroll the subsequent loop
    const int inputChannelsModBlock = inputChannels & (blockDimY - 1);          // reminder calculation using modulo blockDimY operation
    const int sliceCap = inputChannels - inputChannelsModBlock;                 // unrolled loop cap

    // unrolled loop for image fetching and accumulating the partial convolution results
    for (int slice = 0; slice < sliceCap; slice += blockDimY) {
        const int inputChannel = slice + threadIdx.y;
        if (pix < imageSize) {
            // fetching a single quaternion from the input data tensor
            const int i = threadIdx.y * blockDimX + threadIdx.x;                // location of store in shared memory

            const int inputBatchStep = inputChannels * imageSize;
            const int inputChannelStep = inputBatchStep * batchSize;            // helper constant needed because of the channel first format

            const T* iptr = input + inputBatchStep * batchPos + imageSize * inputChannel + pix;     // input[batchPos][inputChannel][pix]
            inputBuffer[i].r = iptr[0];
            inputBuffer[i].i = iptr[1*inputChannelStep];                                            // input[batchPos + batchSize][channel][pix]
            inputBuffer[i].j = iptr[2*inputChannelStep];
            inputBuffer[i].k = iptr[3*inputChannelStep];
        }

        const int weightsChannel = slice + threadIdx.x;
        if (outChannel < outputChannels && weightsChannel < inputChannels) {
            // fetching a single quaternion from the weights tensor
            const int j = threadIdx.x * (blockDimY + bankSkew) + threadIdx.y;   // location of store in shared memory

            const int weightsChannelStep = outputChannels * inputChannels;

            const T* wptr = weights + outChannel * inputChannels + weightsChannel;                    // weights[0][outChannel][weightsChannel]
            weightsBuffer[j].r = wptr[0];
            weightsBuffer[j].i = wptr[1*weightsChannelStep];                                          // weights[1][outChannel][weightsChannel]
            weightsBuffer[j].j = wptr[2*weightsChannelStep];
            weightsBuffer[j].k = wptr[3*weightsChannelStep];
        }

        // wait until all the threads fetch a value from input and weights tensors
        __syncthreads();

        // check if this thread should compute a value in the output tensor
        if (outChannel < outputChannels && pix < imageSize) {
            // calculate the contribution to the final value from the fetched tensor windows
            int indexInput = threadIdx.x;
            int indexWeights = threadIdx.y;

            // unrolling a loop, blockDimY should be divisible by 4, seemingly not much of a performance gain
            #pragma unroll (4)
            for (int i = 0; i < blockDimY; i++, indexInput += blockDimX, indexWeights += blockDimY + bankSkew) {
                const quat<T> ismp = inputBuffer[indexInput];
                const quat<T> wsmp = weightsBuffer[indexWeights];

                // perform regular quaternion mutliplication
                acc.r += ismp.r * wsmp.r - ismp.i * wsmp.i - ismp.j * wsmp.j - ismp.k * wsmp.k;
                acc.i += ismp.r * wsmp.i + ismp.i * wsmp.r + ismp.j * wsmp.k - ismp.k * wsmp.j;
                acc.j += ismp.r * wsmp.j + ismp.j * wsmp.r + ismp.k * wsmp.i - ismp.i * wsmp.k;
                acc.k += ismp.r * wsmp.k + ismp.k * wsmp.r + ismp.i * wsmp.j - ismp.j * wsmp.i;
            }
        }
        // wait until all the threads fetch all the necessary values from shared memory
        __syncthreads();
    }

    // check if some remainder of the loop is to be executed
    // code generally the same as for the main loop
    if (inputChannelsModBlock != 0) {
        const int inputChannel = sliceCap + threadIdx.y;
        if (pix < imageSize && inputChannel < inputChannels) {
            // fetching a single quaternion from the input data tensor
            const int i = threadIdx.y * blockDimX + threadIdx.x;                // location of store in shared memory

            const int inputBatchStep = inputChannels * imageSize;
            const int inputChannelStep = inputBatchStep * batchSize;            // helper constant needed because of the channel first format

            const T* iptr = input + inputBatchStep * batchPos + imageSize * inputChannel + pix;     // input[batchPos][inputChannel][pix]
            inputBuffer[i].r = iptr[0];
            inputBuffer[i].i = iptr[1*inputChannelStep];                                            // input[batchPos + batchSize][channel][pix]
            inputBuffer[i].j = iptr[2*inputChannelStep];
            inputBuffer[i].k = iptr[3*inputChannelStep];
        }

        const int weightsChannel = sliceCap + threadIdx.x;
        if (outChannel < outputChannels && weightsChannel < inputChannels) {
            // fetching a single quaternion from the weights tensor
            const int j = threadIdx.x * (blockDimY + bankSkew) + threadIdx.y;   // location of store in shared memory

            const int weightsChannelStep = outputChannels * inputChannels;

            const T* wptr = weights + outChannel * inputChannels + weightsChannel;                    // weights[0][outChannel][weightsChannel]
            weightsBuffer[j].r = wptr[0];
            weightsBuffer[j].i = wptr[1*weightsChannelStep];                                          // weights[1][outChannel][weightsChannel]
            weightsBuffer[j].j = wptr[2*weightsChannelStep];
            weightsBuffer[j].k = wptr[3*weightsChannelStep];
        }

        // wait until all the threads fetch a value from input and weights tensors
        __syncthreads();

        // check if this thread should compute a value in the output tensor
        if (outChannel < outputChannels && pix < imageSize) {
            // calculate the contribution to the final value from the fetched tensor windows
            int indexInput = threadIdx.x;
            int indexWeights = threadIdx.y;

            for (int i = 0; i < inputChannelsModBlock; i++, indexInput += blockDimX, indexWeights += blockDimY + bankSkew) {
                const quat<T> ismp = inputBuffer[indexInput];
                const quat<T> wsmp = weightsBuffer[indexWeights];

                // perform regular quaternion mutliplication
                acc.r += ismp.r * wsmp.r - ismp.i * wsmp.i - ismp.j * wsmp.j - ismp.k * wsmp.k;
                acc.i += ismp.r * wsmp.i + ismp.i * wsmp.r + ismp.j * wsmp.k - ismp.k * wsmp.j;
                acc.j += ismp.r * wsmp.j + ismp.j * wsmp.r + ismp.k * wsmp.i - ismp.i * wsmp.k;
                acc.k += ismp.r * wsmp.k + ismp.k * wsmp.r + ismp.i * wsmp.j - ismp.j * wsmp.i;
            }
        }
        // no need for the last __syncthreads()
    }

    // check if this thread should compute a value in the output tensor
    if (outChannel < outputChannels && pix < imageSize) {
        const int outputBatchStep = outputChannels * imageSize;
        const int outputChannelStep = outputBatchStep * batchSize;              // helper constant needed because of the channel first format
        const int outputOffset = outputBatchStep * batchPos + imageSize * outChannel + pix;

        // add bias if it is enabled
        if (bias != nullptr) {
            const T* bptr = bias + outChannel;
            acc.r += bptr[0];
            acc.i += bptr[1*outputChannels];
            acc.j += bptr[2*outputChannels];
            acc.k += bptr[3*outputChannels];
        }

        // store the accumulator as a value in the output tensor
        T* optr = output + outputOffset;                                        // output[batchPos][outChannel][pix]
        optr[0] = acc.r;
        optr[1*outputChannelStep] = acc.i;                                      // output[batchPos + batchSize][outChannel][pix]
        optr[2*outputChannelStep] = acc.j;
        optr[3*outputChannelStep] = acc.k;
    }
}


/**
 * @brief Backward pass CUDA kernel for input data gradient computation
 *
 * This kernel works properly only for blockDimX == blockDimY.
 * blockDim.z should be equal to the batch size.
 *
 * With this kernel, each thread computes one input gradient value.
 * Each thread block corresponds to a (blockDimX x blockDimY) patch of the input gradient values.
 *
 * A thread block shares memory, in each input processing step each thread fetches one weights and one total gradient value,
 * and then calculates the contribution of blockDimX (== blockDimY) factors (convolution multiplications and additions)
 * to the final input gradient value.
 *
 * After each input processing step the window of values to fetch moves across the output channels dimension
 * for both weights and total gradient fetching.
 *
 * Intuitively (disregarding batchSize and channel first):
 * Total gradient dimensions: x - imageSize, y - outputChannels
 * Weights dimensions: x - inputChannels, y - outputChannels
 * Input gradient (output) dimensions: x - imageSize, y - inputChannels
 */
template<typename T, unsigned int blockDimX, unsigned int blockDimY>
__global__ void pointwise_conv_grad_input_2D_shared_mem(
    const T* weights, const T* grad, T* output,
    const int batchSize, const int inputChannels, const int outputChannels, const int imageSize
) {
    static_assert(blockDimX == blockDimY, "pointwise_conv_grad_input_2D_shared_mem kernel requires equal x and y thread block dimensions");

    // declare CUDA shared memory
    // no banks skew, as accesses do not cause bank conflicts in shared memory
    __shared__ quat<T> gradBuffer[blockDimX * blockDimY];
    __shared__ quat<T> weightsBuffer[blockDimX * blockDimY];

    // determine the position of the input gradient value this thread corresponds to
    const int inputChannel = blockIdx.y * blockDimY + threadIdx.y;              // output channel
    const int pix = blockIdx.x * blockDimX + threadIdx.x;                       // pixel number in image
    const int batchPos = blockIdx.z;                                            // image number in batch

    // use an accumulator to store the intermediate partial gradient values
    quat<T> acc{ 0, 0, 0, 0 };

    // loop for input tensors fetching and accumulating the partial gradient results
    for (int slice = 0; slice < outputChannels; slice += blockDimY) {
        const int outChannel = slice + threadIdx.y;
        if (outChannel < outputChannels) {
            const int i = threadIdx.y * blockDimX + threadIdx.x;                // location of store in shared memory

            if (pix < imageSize) {
                // fetching a single quaternion from the total gradient tensor
                const int outputBatchStep = outputChannels * imageSize;
                const int outputChannelStep = outputBatchStep * batchSize;      // helper constant needed because of the channel first format

                const T* gptr = grad + outputBatchStep * batchPos + imageSize * outChannel + pix;   // grad[batchPos][outChannel][pix]
                gradBuffer[i].r = gptr[0];
                gradBuffer[i].i = gptr[1*outputChannelStep];                                        // grad[batchPos + batchSize][outChannel][pix]
                gradBuffer[i].j = gptr[2*outputChannelStep];
                gradBuffer[i].k = gptr[3*outputChannelStep];
            }

            const int weightsChannel = blockIdx.y * blockDimY + threadIdx.x;
            if (weightsChannel < inputChannels) {
                // fetching a single quaternion from the weights tensor
                const int weightsChannelStep = outputChannels * inputChannels;

                const T* wptr = weights + outChannel * inputChannels + weightsChannel;              // weights[0][outChannel][weightsChannel]
                weightsBuffer[i].r = wptr[0];
                weightsBuffer[i].i = wptr[1*weightsChannelStep];                                    // weights[1][outChannel][weightsChannel]
                weightsBuffer[i].j = wptr[2*weightsChannelStep];
                weightsBuffer[i].k = wptr[3*weightsChannelStep];
            }
        }

        // wait until all the threads fetch a value from total gradient and weights tensors
        __syncthreads();

        // check if this thread should compute a value in the input gradient tensor
        if (inputChannel < inputChannels && pix < imageSize) {
            // calculate the contribution to the final value from the fetched tensor windows
            const int cap = min(outputChannels - slice, blockDimY);
            int index = 0;

            for (int i = 0; i < cap; i++, index += blockDimX) {
                const quat<T> gsmp = gradBuffer[index + threadIdx.x];
                const quat<T> wsmp = weightsBuffer[index + threadIdx.y];

                // perform regular quaternion mutliplication
                acc.r += + gsmp.r * wsmp.r + gsmp.i * wsmp.i + gsmp.j * wsmp.j + gsmp.k * wsmp.k;
                acc.i += - gsmp.r * wsmp.i + gsmp.i * wsmp.r - gsmp.j * wsmp.k + gsmp.k * wsmp.j;
                acc.j += - gsmp.r * wsmp.j + gsmp.i * wsmp.k + gsmp.j * wsmp.r - gsmp.k * wsmp.i;
                acc.k += - gsmp.r * wsmp.k - gsmp.i * wsmp.j + gsmp.j * wsmp.i + gsmp.k * wsmp.r;
            }
        }
        // wait until all the threads fetch all the necessary values from shared memory
        __syncthreads();
    }

    // check if this thread should compute a value in the input gradient tensor
    if (inputChannel < inputChannels && pix < imageSize) {
        const int outputBatchStep = inputChannels * imageSize;
        const int outputChannelStep = outputBatchStep * batchSize;                                  // helper constant needed because of the channel first format
        const int outputOffset = outputBatchStep * batchPos + imageSize * inputChannel + pix;

        // store the accumulator as a value in the input gradient tensor
        T* optr = output + outputOffset;                                                            // output[batchPos][inputChannel][pix]
        optr[0] = acc.r;
        optr[1*outputChannelStep] = acc.i;                                                          // output[batchPos + batchSize][inputChannel][pix]
        optr[2*outputChannelStep] = acc.j;
        optr[3*outputChannelStep] = acc.k;
    }
}


/**
 * @brief Backward pass CUDA kernel for weights gradient computation
 *
 * This kernel works properly only for blockDimX == blockDimY.
 * blockDim.z should be equal to 1.
 *
 * With this kernel, each thread with (threadIdx.z == 0) computes one weights gradient value.
 * Threads with other threadIdx.z values compute partial results, which are eventually accumulated
 * by the corresponding threads with (threadIdx.z == 0)
 *
 * Each thread block corresponds to a (blockDimX x blockDimY) patch of the weights gradient values.
 *
 * A thread block shares memory, in each input processing step each thread fetches one input and one total gradient value,
 * and then calculates the contribution of blockDimX (== blockDimY) factors (convolution multiplications and additions)
 * to the final weights gradient value.
 *
 * The input processing consists of two loops, in the outer one the window of values moves across the batch size
 * dimension of the tensors, the inner one the window moves across the imageSize dimension of the tensors.
 *
 * Intuitively (disregarding batchSize and channel first):
 * Total gradient dimensions: x - imageSize, y - outputChannels
 * Input dimensions: x - imageSize, y - inputChannels
 * Weights gradient (output) dimensions: x - inputChannels, y - outputChannels
 */
template<typename T, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int bankSkew>
__global__ void pointwise_conv_grad_weights_3D_thread_block(
    const T* input, const T* grad, T* output,
    const int batchSize, const int inputChannels, const int outputChannels, const int imageSize
) {
    static_assert(blockDimX == blockDimY, "pointwise_conv_grad_weights_3D_thread_block kernel requires equal x and y thread block dimensions");

    // declare CUDA shared memory
    // use banks skew to avoid bank conflicts in shared memory (seemingly not much of a difference though)
    __shared__ quat<T> inputBuffer[(blockDimX + bankSkew) * blockDimY * blockDimZ];
    __shared__ quat<T> gradBuffer[(blockDimX + bankSkew) * blockDimY * blockDimZ];

    // determine the position of the weights gradient value this thread corresponds to
    const int inputChannel = blockIdx.x * blockDimX + threadIdx.x;
    const int outputChannel = blockIdx.y * blockDimY + threadIdx.y;

    // use an accumulator to store the intermediate partial gradient values
    quat<T> acc { 0, 0, 0, 0 };

    // loops for input tensors fetching and accumulating the partial gradient results
    for (int sliceBatch = 0; sliceBatch < batchSize; sliceBatch += blockDimZ) {
        for (int sliceImage = 0; sliceImage < imageSize; sliceImage += blockDimX) {

            // calculate the position across the imageSize and batchSize dimensions of the quaternion to fetch
            const int posInImage = sliceImage + threadIdx.x;
            const int posInBatch = sliceBatch + threadIdx.z;
            if (posInImage < imageSize && posInBatch < batchSize) {

                const int i = threadIdx.z * (blockDimX + bankSkew) * blockDimY
                    + threadIdx.x * (blockDimX + bankSkew) + threadIdx.y;                           // location of store in shared memory

                const int fetchedInputChannel = blockIdx.x * blockDimX + threadIdx.y;
                if (fetchedInputChannel < inputChannels) {
                    // fetching a single quaternion from the input data tensor

                    const int inputBatchStep = inputChannels * imageSize;
                    const int inputChannelStep = inputBatchStep * batchSize;                        // helper constant needed because of the channel first format

                    auto iptr = input + inputBatchStep * posInBatch + imageSize * fetchedInputChannel + posInImage;     // input[posInBatch][fetchedInputChannel][posInImage]
                    inputBuffer[i].r = iptr[0];
                    inputBuffer[i].i = iptr[1 * inputChannelStep];                                          // input[posInBatch + batchSize][fetchedInputChannel][posInImage]
                    inputBuffer[i].j = iptr[2 * inputChannelStep];
                    inputBuffer[i].k = iptr[3 * inputChannelStep];
                }

                if (outputChannel < outputChannels) {
                    // fetching a single quaternion from the total gradient tensor

                    const int gradBatchStep = outputChannels * imageSize;
                    const int gradChannelStep = gradBatchStep * batchSize;                          // helper constant needed because of the channel first format

                    auto gptr = grad + gradBatchStep * posInBatch + imageSize * outputChannel + posInImage;     // grad[posInBatch][outputChannel][posInImage]
                    gradBuffer[i].r = gptr[0];
                    gradBuffer[i].i = gptr[1 * gradChannelStep];                                    // grad[posInBatch + batchSize][outputChannel][posInImage]
                    gradBuffer[i].j = gptr[2 * gradChannelStep];
                    gradBuffer[i].k = gptr[3 * gradChannelStep];
                }
            }

            // wait until all the threads fetch a value from total gradient and input tensors
            __syncthreads();

            // check if this thread should compute a value in the weights gradient tensor
            if (inputChannel < inputChannels && outputChannel < outputChannels && posInBatch < batchSize) {
                // calculate the contribution to the final value from the fetched tensor windows
                const int cap = min(imageSize - sliceImage, blockDimX);
                int index = threadIdx.z * (blockDimX + bankSkew) * blockDimY;

                for (int i = 0; i < cap; i++, index += blockDimX + bankSkew) {
                    quat<T> ismp = inputBuffer[index + threadIdx.x];
                    quat<T> gsmp = gradBuffer[index + threadIdx.y];

                    // perform regular quaternion mutliplication
                    acc.r += + gsmp.r * ismp.r + gsmp.i * ismp.i + gsmp.j * ismp.j + gsmp.k * ismp.k;
                    acc.i += - gsmp.r * ismp.i + gsmp.i * ismp.r + gsmp.j * ismp.k - gsmp.k * ismp.j;
                    acc.j += - gsmp.r * ismp.j - gsmp.i * ismp.k + gsmp.j * ismp.r + gsmp.k * ismp.i;
                    acc.k += - gsmp.r * ismp.k + gsmp.i * ismp.j - gsmp.j * ismp.i + gsmp.k * ismp.r;
                }
            }
            // wait until all the threads fetch all the necessary values from shared memory
            __syncthreads();
        }
    }

    // store intermediate weights gradient values for threads with (threadIdx.z == 0) to collect
    inputBuffer[threadIdx.z * blockDimX * blockDimY + threadIdx.y * blockDimX + threadIdx.x] = acc;

    // wait until all threads store their accumulators in shared memory
    __syncthreads();

    // check if this thread should compute a value in the weights gradient tensor
    if (threadIdx.z == 0 && inputChannel < inputChannels && outputChannel < outputChannels) {

        // accumulate the values computed by threads with different threadIdx.z
        // could use warp-level primitive __shfl_down_sync() for it
        for (int i = 1; i < blockDimZ; i++) {
            const quat<T> smp = inputBuffer[i * blockDimX * blockDimY + threadIdx.y * blockDimX + threadIdx.x];
            acc.r += smp.r;
            acc.i += smp.i;
            acc.j += smp.j;
            acc.k += smp.k;
        }

        const int finalChannelStep = inputChannels * outputChannels;                                // helper constant needed because of the channel first format

        // store the accumulator as a value in the weights gradient tensor
        auto optr = output + outputChannel * inputChannels + inputChannel;                          // output[0][outputChannel][inputChannel]
        optr[0] = acc.r;
        optr[1 * finalChannelStep] = acc.i;                                                         // output[1][outputChannel][inputChannel]
        optr[2 * finalChannelStep] = acc.j;
        optr[3 * finalChannelStep] = acc.k;

    }
}


/**
 * @brief Backward pass CUDA kernel for weights gradient computation
 *
 * This kernel works properly only for blockDimX == blockDimY.
 * blockDim.z should be equal to 1.
 * inputChannels should be divisible by inpChansServed.
 * outputChannels should be divisible by outChansServed.
 *
 * With this kernel, each thread block computes a patch of EXACTLY (inpChansServed x outChansServed)
 * values in the weights gradient tensor.
 *
 * Each thread in a block keeps (inpChansServed x outChansServed) accumulators, each corresponding to a different
 * value in the weights gradient tensor. After all the input is processed, threads with (threadIdx.y == 0)
 * add up all the accumulators from threads with other threadIdx.y, and consequently threads with (threadIdx.x == 0 && threadIdx.y == 0)
 * add up all the accumulators from threads with (threadIdx.y == 0). After this two-step accumulation,
 * final weights gradient values are stored in the corresponding tensor.
 *
 * Threads in a block use the shared memory only for the final two-step accumulation.
 * This kernel allows for larger grid sizes than pointwise_conv_grad_weights_3D_thread_block, and keeps the accumulators in the registers.
 * Too large of a inpChansServed/outChansServed parameter and the accumulators will spill from the registers to a very slow local memory.
 *
 * In each input processing step each thread fetches inpChansServed input and outChansServed total gradient value,
 * and then calculates the contribution of (inpChansServed x outChansServed) factors (convolution multiplications and additions)
 * to the final weights gradient value.
 *
 * The input processing consists of two loops, in the outer one the window of values moves across the batch size
 * dimension of the tensors, the inner one the window moves across the imageSize dimension of the tensors.
 *
 * Intuitively (disregarding batchSize and channel first):
 * Total gradient dimensions: x - imageSize, y - outputChannels
 * Input dimensions: x - imageSize, y - inputChannels
 * Weights gradient (output) dimensions: x - inputChannels, y - outputChannels
 */
template<typename T, unsigned int blockDimX, unsigned int blockDimY,
        unsigned int inpChansServed, unsigned int outChansServed>
__global__ void pointwise_conv_grad_weights_acc_in_regs(const T* input, const T* grad, T* output,
        const int batchSize, const int inputChannels, const int outputChannels, const int imageSize) {
    __shared__ quat<T> accumulators[blockDimX * blockDimY];

    // determine the position of the 'top-left' weights gradient value to compute
    const int firstInputChannel = inpChansServed * blockIdx.x;
    const int firstOutputChannel = outChansServed * blockIdx.y;

    // compute the helper constants needed because of the channel first format
    const int inputBatchStep = inputChannels * imageSize;
    const int inputChannelStep = inputBatchStep * batchSize;

    const int outputBatchStep = outputChannels * imageSize;
    const int outputChannelStep = outputBatchStep * batchSize;

    // use several accumulators to store the intermediate partial gradient values for different final values
    quat<T> acc[inpChansServed * outChansServed] = { 0, 0, 0, 0 };

    // loops for input tensors fetching and accumulating the partial gradient results
    for (int sliceBatch = 0; sliceBatch < batchSize; sliceBatch += blockDimY) {
        for (int sliceImage = 0; sliceImage < imageSize; sliceImage += blockDimX) {

            // calculate the position across the imageSize and batchSize dimensions of the quaternion to fetch
            const int posInImage = sliceImage + threadIdx.x;
            const int posInBatch = sliceBatch + threadIdx.y;

            if (posInImage < imageSize && posInBatch < batchSize) {

                quat<T> ismps[inpChansServed] = { 0, 0, 0, 0 };
                quat<T> gsmps[outChansServed] = { 0, 0, 0, 0 };

                // fetching EXACTLY inpChansServed quaternions from the input data tensor
                for (int inpChan = 0; inpChan < inpChansServed; inpChan++) {
                    const int inputChannel = firstInputChannel + inpChan;

                    auto iptr = input + inputBatchStep * posInBatch + imageSize * inputChannel + posInImage;        // input[posInBatch][inputChannel][posInImage]
                    ismps[inpChan].r = iptr[0];
                    ismps[inpChan].i = iptr[1*inputChannelStep];                                                    // input[posInBatch + batchSize][inputChannel][posInImage]
                    ismps[inpChan].j = iptr[2*inputChannelStep];
                    ismps[inpChan].k = iptr[3*inputChannelStep];
                }

                // fetching EXACTLY outChansServed quaternions from the input data tensor
                for (int outChan = 0; outChan < outChansServed; outChan++) {
                    const int outputChannel = firstOutputChannel + outChan;

                    auto gptr = grad + outputBatchStep * posInBatch + imageSize * outputChannel + posInImage;       // grad[posInBatch][outputChannel][posInImage]
                    gsmps[outChan].r = gptr[0];
                    gsmps[outChan].i = gptr[1*outputChannelStep];                                                   // grad[posInBatch + batchSize][outputChannel][posInImage]
                    gsmps[outChan].j = gptr[2*outputChannelStep];
                    gsmps[outChan].k = gptr[3*outputChannelStep];
                }

                // calculate the contribution to the final values from the fetched tensor windows
                for (unsigned int outChan = 0; outChan < outChansServed; outChan++) {
                    for (unsigned int inpChan = 0; inpChan < inpChansServed; inpChan++) {
                        const int whichAcc = outChan * inpChansServed + inpChan;

                        const quat<T> ismp = ismps[inpChan];
                        const quat<T> gsmp = gsmps[outChan];

                        // perform regular quaternion mutliplication
                        acc[whichAcc].r += + gsmp.r * ismp.r + gsmp.i * ismp.i + gsmp.j * ismp.j + gsmp.k * ismp.k;
                        acc[whichAcc].i += - gsmp.r * ismp.i + gsmp.i * ismp.r + gsmp.j * ismp.k - gsmp.k * ismp.j;
                        acc[whichAcc].j += - gsmp.r * ismp.j - gsmp.i * ismp.k + gsmp.j * ismp.r + gsmp.k * ismp.i;
                        acc[whichAcc].k += - gsmp.r * ismp.k + gsmp.i * ismp.j - gsmp.j * ismp.i + gsmp.k * ismp.r;
                    }
                }
            }
        }
    }

    const int posInBlock = threadIdx.y * blockDimX + threadIdx.x;                                   // location of store in shared memory
    // loop for each of the (inpChansServed x outChansServed) output values
    for (unsigned int outChan = 0; outChan < outChansServed; outChan++) {
        for (unsigned int inpChan = 0; inpChan < inpChansServed; inpChan++) {

            // store accumulator in shared memory
            const int whichAcc = outChan * inpChansServed + inpChan;
            accumulators[posInBlock] = acc[whichAcc];
            // wait until all threads store the accumulator
            __syncthreads();

            if (threadIdx.y == 0) {
                // accumulation through columns in thread block
                for (int i = 1; i < blockDimY; i++) {
                    const int posInShared = i * blockDimX + threadIdx.x;
                    const quat<T> smp = accumulators[posInShared];
                    acc[whichAcc].r += smp.r;
                    acc[whichAcc].i += smp.i;
                    acc[whichAcc].j += smp.j;
                    acc[whichAcc].k += smp.k;
                }

                // store column-aggregated accumulator in shared memory
                accumulators[threadIdx.x] = acc[whichAcc];
            }
            // wait until all columns are aggregated
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                // accumulation of column-aggregated values through the first row in thread block
                for (int i = 1; i < blockDimX; i++) {
                    const int posInShared = i;
                    const quat<T> smp = accumulators[posInShared];
                    acc[whichAcc].r += smp.r;
                    acc[whichAcc].i += smp.i;
                    acc[whichAcc].j += smp.j;
                    acc[whichAcc].k += smp.k;
                }

                const int inputChannel = firstInputChannel + inpChan;
                const int outputChannel = firstOutputChannel + outChan;
                const int finalChannelStep = inputChannels * outputChannels;                        // helper constant needed because of the channel first format

                // store the aggregated accumulator as a value in the weights gradient tensor
                auto optr = output + outputChannel * inputChannels + inputChannel;                  // output[0][outputChannel][inputChannel]
                optr[0] = acc[whichAcc].r;
                optr[1 * finalChannelStep] = acc[whichAcc].i;                                       // output[1][outputChannel][inputChannel]
                optr[2 * finalChannelStep] = acc[whichAcc].j;
                optr[3 * finalChannelStep] = acc[whichAcc].k;
            }
            // wait until all the values are aggregated
            __syncthreads();
        }
    }
}


// 'using' declarations for better code readability and partial template specialization
template<typename T>
using ForwardFunctor = QuatKernelPointwiseConvForwardFunctor<device::CUDA, T>;

template<typename T>
using BackwardFunctor = QuatKernelPointwiseConvBackwardFunctor<device::CUDA, T>;

template<typename T>
using ForwardKernelPtr = void (*) (const T*, const T*, const T*, T*, const int, const int, const int, const int);

template<typename T>
using BackwardKernelPtr = void (*) (const T*, const T*, T*, const int, const int, const int, const int);


// Utility functions

/**
 * @brief Compare two CUDA dim3 structs
 *
 * @return true if structs are equal memberwise
 */
inline bool sameDim3(const dim3 &a, const dim3 &b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}


/**
 * @brief Assert that the thread block x and y dimensions are equal
 *
 * @param conf                          kernel configuration with the thread block to check
 */
inline void assertEqualDimXDimY(const ConvKernelConfiguration& conf) {
    if (conf.threads.x != conf.threads.y) {
        const std::string error_msg = "Thread block x and y dimensions must be equal for the chosen kernel: " + conf.toString();
        throw std::invalid_argument(error_msg);
    }
}


/**
 * @brief Assert that a kernel was set
 *
 * @param kernelSet                     control boolean
 * @param conf                          kernel configuration to print in the error message
 */
inline void assertKernelSet(bool kernelSet, const ConvKernelConfiguration& conf) {
    if (!kernelSet) {
        const std::string error_msg = "Unhandled kernel configuration: " + conf.toString();
        throw std::invalid_argument(error_msg);
    }
}


/**
 * @brief Check if the number of registers per thread block is enough for the chosen thread block size
 *
 * Currently used kernel threads often use between 32 and 64 registers, as checked with Nsight Compute.
 * Running kernels using too many registers causes the 'too many resources requested for launch' CUDA error.
 *
 * @param threads                       thread block dimensions
 * @param registersPerThreadBlock       number of available registers per thread block on the current device
 */
inline bool checkEnoughRegistersForLaunch(const dim3& threads, int registersPerThreadBlock) {
    const int threadBlockSize = threads.x * threads.y * threads.z;
    const int MAX_REGISTER_USAGE = 64;
    return ((threadBlockSize * MAX_REGISTER_USAGE) <= registersPerThreadBlock);
}


// Utility functions for Forward Functor

/**
 * @brief Utility parser for the pointwiseForward_2DSharedMemory kernel
 *
 * @tparam T                            scalar datatype
 * @tparam blockDimXY                   thread block x (== y) dimension
 * @tparam bankSkew                     bank skew parameter to be set
 *
 * @param conf                          kernel configuration to be parsed
 * @param kernel                        kernel pointer set if the kernel configuration matches template parameters
 * @return                              true if the kernel is set
 */
template<typename T, unsigned int blockDimXY, unsigned int bankSkew>
bool trySetKernelForward2DSharedMemory(
    const ConvKernelConfiguration& conf, ForwardKernelPtr<T>& kernel
) {
    if (sameDim3(conf.threads, dim3{blockDimXY, blockDimXY, 1})) {
        kernel = pointwise_conv_forward_2D_shared_mem<T, blockDimXY, blockDimXY, bankSkew>;
        return true;
    }
    return false;
}


// Utility functions for Backward Functor

/**
 * @brief Utility parser for the pointwiseInputGrad_2DSharedMemory kernel
 *
 * @tparam T                            scalar datatype
 * @tparam blockDimXY                   thread block x (== y) dimension
 *
 * @param conf                          kernel configuration to be parsed
 * @param kernel                        kernel pointer set if the kernel configuration matches template parameters
 * @return                              true if the kernel is set
 */
template<typename T, unsigned int blockDimXY>
bool trySetKernelInputGrad2DSharedMemory(
    const ConvKernelConfiguration& conf, BackwardKernelPtr<T>& kernel
) {
    if (sameDim3(conf.threads, dim3{blockDimXY, blockDimXY, 1})) {
        kernel = pointwise_conv_grad_input_2D_shared_mem<T, blockDimXY, blockDimXY>;
        return true;
    }
    return false;
}


/**
 * @brief Utility parser for the pointwiseWeightsGrad_3DThreadBlock kernel
 *
 * @tparam T                            scalar datatype
 * @tparam blockDimXY                   thread block x (== y) dimension
 * @tparam blockDimZ                    thread block z dimension
 * @tparam bankSkew                     bank skew parameter to be set
 *
 * @param conf                          kernel configuration to be parsed
 * @param kernel                        kernel pointer set if the kernel configuration matches template parameters
 * @return                              true if the kernel is set
 */
template<typename T, unsigned int blockDimXY, unsigned int blockDimZ, unsigned int bankSkew>
bool trySetKernelWeightsGrad3DThreadBlock(
    const ConvKernelConfiguration& conf, BackwardKernelPtr<T>& kernel
) {
    if (sameDim3(conf.threads, dim3{blockDimXY, blockDimXY, blockDimZ})) {
        kernel = pointwise_conv_grad_weights_3D_thread_block<T, blockDimXY, blockDimXY, blockDimZ, bankSkew>;
        return true;
    }
    return false;
}


/**
 * @brief Utility parser helper for the pointwiseWeightsGrad_AccumulatorsInRegisters kernel
 *
 * @tparam T                            scalar datatype
 * @tparam configFirst                  first parameter of ConvKernelConfiguration.config, matched with conf.config.first
 * @tparam configSecond                 second parameter of ConvKernelConfiguration.config, matched with conf.config.second
 * @tparam blockDimX                    thread block x dimension
 * @tparam blockDimY                    thread block y dimension
 *
 * @param conf                          kernel configuration to be parsed
 * @param kernel                        kernel pointer set if the kernel configuration matches template parameters
 * @return                              true if the kernel is set
 */
template <typename T, int configFirst, int configSecond, unsigned int blockDimX, unsigned int blockDimY>
bool trySetKernelWeightsGradAccumulatorsInRegistersHelper(
    const ConvKernelConfiguration& conf, BackwardKernelPtr<T>& kernel
) {
    if (sameDim3(conf.threads, dim3{blockDimX, blockDimY, 1})) {
        kernel = pointwise_conv_grad_weights_acc_in_regs<T, blockDimX, blockDimY, configFirst, configSecond>;
        return true;
    }
    return false;
}


/**
 * @brief Utility parser for the pointwiseWeightsGrad_AccumulatorsInRegisters kernel
 *
 * @tparam T                            scalar datatype
 * @tparam configFirst                  first parameter of ConvKernelConfiguration.config
 * @tparam configSecond                 second parameter of ConvKernelConfiguration.config
 *
 * @param conf                          kernel configuration to be parsed
 * @param kernel                        kernel pointer set if the kernel configuration matches template parameters
 * @return                              true if the kernel is set
 */
template<typename T, int configFirst, int configSecond>
bool trySetKernelWeightsGradAccumulatorsInRegisters(
    const ConvKernelConfiguration& conf, BackwardKernelPtr<T>& kernel
) {
    bool kernelSet {false};
    if (conf.config.first == configFirst && conf.config.second == configSecond) {
        kernelSet =
            trySetKernelWeightsGradAccumulatorsInRegistersHelper<T, configFirst, configSecond, 32, 32>(conf, kernel)
            || trySetKernelWeightsGradAccumulatorsInRegistersHelper<T, configFirst, configSecond, 32, 16>(conf, kernel);
    }
    return kernelSet;
}


// Forward Functor

// ForwardFunctor<T> doesn't work here or with other definitions (but works with declarations)
template<typename T>
void QuatKernelPointwiseConvForwardFunctor<device::CUDA, T>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
) {
    const auto biasDataPtr = tensors.inputTensor3 != nullptr ? tensors.inputTensor3->getDataPtr() : nullptr;

    kernelPack.kernel<<<kernelPack.blocks, kernelPack.threads, 0, tensors.inputTensor1.getDevice().stream()>>>(
        tensors.inputTensor1.getDataPtr(), tensors.inputTensor2.getDataPtr(), biasDataPtr, tensors.outputTensor.getDataPtr(),
        convDesc.batchSize, convDesc.inputChannels,
        convDesc.outputChannels, convDesc.imageSize
    );
    cudnn::Context::raiseIfError("ForwardFunctor launchKernel failed");
}


template<typename T>
bool QuatKernelPointwiseConvForwardFunctor<device::CUDA, T>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
) {
    ForwardKernelPtr kernel;
    dim3 blocks;
    dim3 threads = conf.threads;
    bool kernelSet {false};

    if (!checkEnoughRegistersForLaunch(threads, this->registersPerThreadBlock)) {
        return false;
    }

    switch (conf.kernel) {
    case ConvKernelType::skip:
        return false;

    case ConvKernelType::pointwiseForward_2DSharedMemory:
        blocks = dim3(
            ceili(convDesc.imageSize, threads.x),
            ceili(convDesc.outputChannels, threads.y),
            convDesc.batchSize);

#ifdef UPSTRIDE_DEBUG
        assertEqualDimXDimY(conf);
#endif
        kernelSet =
            trySetKernelForward2DSharedMemory<T, 32, 1>(conf, kernel)
            || trySetKernelForward2DSharedMemory<T, 16, 1>(conf, kernel)
            || trySetKernelForward2DSharedMemory<T, 8, 1>(conf, kernel);

        break;

    default:
        throw std::invalid_argument("Invalid kernel type chosen for forward convolution");
    }

#ifdef UPSTRIDE_DEBUG
    assertKernelSet(kernelSet, conf);
#endif
    // only pack the kernel if all the setup was done correctly
    kernelPack = KernelPack{kernel, blocks, threads};
    return true;
}


// Backward Functor

template<typename T>
bool QuatKernelPointwiseConvBackwardFunctor<device::CUDA, T>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
) {
    BackwardKernelPtr kernel;
    dim3 blocks;
    dim3 threads = conf.threads;
    bool kernelSet {false};

    if (!checkEnoughRegistersForLaunch(threads, this->registersPerThreadBlock)) {
        return false;
    }

    switch(conf.kernel) {
    case ConvKernelType::skip:
        return false;

    case ConvKernelType::pointwiseInputGrad_2DSharedMemory:
        blocks = dim3(
            ceili(convDesc.imageSize, threads.x),
            ceili(convDesc.inputChannels, threads.y),
            convDesc.batchSize);

#ifdef UPSTRIDE_DEBUG
        assertEqualDimXDimY(conf);
#endif
        kernelSet =
            trySetKernelInputGrad2DSharedMemory<T, 32>(conf, kernel)
            || trySetKernelInputGrad2DSharedMemory<T, 16>(conf, kernel)
            || trySetKernelInputGrad2DSharedMemory<T, 8>(conf, kernel);

        break;

    case ConvKernelType::pointwiseWeightsGrad_3DThreadBlock:
        blocks = dim3(
            ceili(convDesc.inputChannels, threads.x),
            ceili(convDesc.outputChannels, threads.y),
            1);

#ifdef UPSTRIDE_DEBUG
        assertEqualDimXDimY(conf);
#endif
        kernelSet =
            trySetKernelWeightsGrad3DThreadBlock<T, 32, 1, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 16, 4, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 16, 2, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 8, 16, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 8, 8, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 4, 64, 1>(conf, kernel)
            || trySetKernelWeightsGrad3DThreadBlock<T, 4, 32, 1>(conf, kernel);

        break;

    case ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters:
        if (convDesc.inputChannels % conf.config.first != 0
            || convDesc.outputChannels % conf.config.second != 0
        ) {
            // with this kernel, each thread is responsible for a block of EXACTLY (config.first x config.second) values
            return false;
        }

        blocks = dim3(
            convDesc.inputChannels / conf.config.first,
            convDesc.outputChannels / conf.config.second,
            1);

        kernelSet =
            trySetKernelWeightsGradAccumulatorsInRegisters<T, 2, 2>(conf, kernel)
            || trySetKernelWeightsGradAccumulatorsInRegisters<T, 2, 1>(conf, kernel)
            || trySetKernelWeightsGradAccumulatorsInRegisters<T, 1, 2>(conf, kernel)
            || trySetKernelWeightsGradAccumulatorsInRegisters<T, 1, 1>(conf, kernel);

        break;

    default:
        throw std::invalid_argument("Invalid kernel type chosen for convolution gradient computation");
    }

#ifdef UPSTRIDE_DEBUG
    assertKernelSet(kernelSet, conf);
#endif
    // only pack the kernel if all the setup was done correctly
    kernelPack = KernelPack{kernel, blocks, threads};
    return true;
}


template<typename T>
void QuatKernelPointwiseConvBackwardFunctor<device::CUDA, T>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
) {
    kernelPack.kernel<<<kernelPack.blocks, kernelPack.threads, 0, tensors.inputTensor1.getDevice().stream()>>>(
        tensors.inputTensor1.getDataPtr(),  tensors.inputTensor2.getDataPtr(),  tensors.outputTensor.getDataPtr(),
        convDesc.batchSize, convDesc.inputChannels,
        convDesc.outputChannels, convDesc.imageSize
    );
    cudnn::Context::raiseIfError("BackwardFunctor launchKernel failed");
}


// Forward declarations with specialized templates

// Forward Functor

template void ForwardFunctor<float>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
);

template bool ForwardFunctor<float>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
);

// Backward Functor

template void BackwardFunctor<float>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
);

template bool BackwardFunctor<float>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
);

#ifdef UPSTRIDE_ENABLE_FP16

// Forward Functor

template void ForwardFunctor<half>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
);

template bool ForwardFunctor<half>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
);

// Backward Functor

template void BackwardFunctor<half>::launchKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
);

template bool BackwardFunctor<half>::interpretKernelConf(
    const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
);

#endif

}
}