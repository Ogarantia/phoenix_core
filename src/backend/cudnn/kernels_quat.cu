#include <cuda.h>

#include "kernels.hpp"
#include "tensor.hpp"

static const int NUM_THREADS = 1024;  //!< default number of threads


/**
 * @brief Implementations of CUDA kernels performing quaternion compositions into 8 scalar lanes
 * Refer to TensorManipulations for more details.
 */
namespace kernels {

template <typename T>
__global__ void decomposeLeftInput(const T* input0, const T* input1, const T* input2, const T* input3,
                                   T* output0, T* output1, T* output2, T* output3, T* output4, T* output5, T* output6, T* output7,
                                   int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        output0[i] = input3[i] + input1[i];
        output1[i] = input0[i] - input2[i];
        output2[i] = input0[i] + input2[i];
        output3[i] = input3[i] - input1[i];
        output4[i] = input3[i] - input2[i];
        output5[i] = input1[i] + input0[i];
        output6[i] = input0[i] - input1[i];
        output7[i] = input3[i] + input2[i];
    }
}

template <typename T>
__global__ void decomposeRightInput(const T* input0, const T* input1, const T* input2, const T* input3,
                                    T* output0, T* output1, T* output2, T* output3, T* output4, T* output5, T* output6, T* output7,
                                    int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        output0[i] = input1[i] + input2[i];
        output1[i] = input0[i] + input3[i];
        output2[i] = input0[i] - input3[i];
        output3[i] = input1[i] - input2[i];
        output4[i] = input2[i] - input3[i];
        output5[i] = input1[i] + input0[i];
        output6[i] = input2[i] + input3[i];
        output7[i] = input0[i] - input1[i];
    }
}

template <typename T>
__global__ void decomposeOutputGrad(const T* input0, const T* input1, const T* input2, const T* input3,
                                    T* output0, T* output1, T* output2, T* output3, T* output4, T* output5, T* output6, T* output7,
                                    int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        const T t1 = input0[i] + input1[i];
        const T t3 = input0[i] - input1[i];
        const T t2 = input2[i] + input3[i];
        const T t4 = input2[i] - input3[i];
        output0[i] = (T).5 * (t2 - t1);
        output1[i] = (T).5 * (t3 - t4);
        output2[i] = (T).5 * (t3 + t4);
        output3[i] = (T).5 * (t1 + t2);
        output4[i] = input0[i];
        output5[i] = input1[i];
        output6[i] = input2[i];
        output7[i] = input3[i];
    }
}

template <typename T>
__global__ void recomposeOutput(const T* input0, const T* input1, const T* input2, const T* input3, const T* input4, const T* input5, const T* input6, const T* input7,
                                T* output0, T* output1, T* output2, T* output3,
                                int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        const T a2 = input0[i] + input1[i] + input2[i];
        const T a5 = (T).5 * (a2 + input3[i]);
        output0[i] = a5 - input0[i] + input4[i];
        output1[i] = a5 - a2 + input5[i];
        output2[i] = a5 - input1[i] + input6[i];
        output3[i] = a5 - input2[i] + input7[i];
    }
}

template <typename T>
__global__ void recomposeLeftInputGrad(const T* input0, const T* input1, const T* input2, const T* input3, const T* input4, const T* input5, const T* input6, const T* input7,
                                       T* output0, T* output1, T* output2, T* output3,
                                       int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        output0[i] = input1[i] + input2[i] + input5[i] + input6[i];
        output1[i] = input0[i] - input3[i] + input5[i] - input6[i];
        output2[i] = input2[i] - input1[i] - input4[i] + input7[i];
        output3[i] = input0[i] + input3[i] + input4[i] + input7[i];
    }
}

template <typename T>
__global__ void recomposeRightInputGrad(const T* input0, const T* input1, const T* input2, const T* input3, const T* input4, const T* input5, const T* input6, const T* input7,
                                        T* output0, T* output1, T* output2, T* output3,
                                        int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        output0[i] = input1[i] + input2[i] + input5[i] + input7[i];
        output1[i] = input0[i] + input3[i] + input5[i] - input7[i];
        output2[i] = input0[i] - input3[i] + input4[i] + input6[i];
        output3[i] = input1[i] - input2[i] - input4[i] + input6[i];
    }
}
}  // namespace kernels


using namespace upstride;

template <typename T>
void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const T, 4>& inLeft, AllocatedTensor<device::CUDA, T>* outLeft[8],
                               const TensorSplit<device::CUDA, const T, 4>& inRight, AllocatedTensor<device::CUDA, T>* outRight[8]) {
    const int leftLen = inLeft.shape().numel();
    ::kernels::decomposeLeftInput<<<ceili(leftLen, NUM_THREADS), NUM_THREADS, 0, inLeft[0].getDevice().stream()>>>(
        inLeft[0].getDataPtr(), inLeft[1].getDataPtr(), inLeft[2].getDataPtr(), inLeft[3].getDataPtr(),
        outLeft[0]->getDataPtr(), outLeft[1]->getDataPtr(), outLeft[2]->getDataPtr(), outLeft[3]->getDataPtr(),
        outLeft[4]->getDataPtr(), outLeft[5]->getDataPtr(), outLeft[6]->getDataPtr(), outLeft[7]->getDataPtr(),
        leftLen);
    const int rightLen = inRight.shape().numel();
    ::kernels::decomposeRightInput<<<ceili(rightLen, NUM_THREADS), NUM_THREADS, 0, inRight[0].getDevice().stream()>>>(
        inRight[0].getDataPtr(), inRight[1].getDataPtr(), inRight[2].getDataPtr(), inRight[3].getDataPtr(),
        outRight[0]->getDataPtr(), outRight[1]->getDataPtr(), outRight[2]->getDataPtr(), outRight[3]->getDataPtr(),
        outRight[4]->getDataPtr(), outRight[5]->getDataPtr(), outRight[6]->getDataPtr(), outRight[7]->getDataPtr(),
        rightLen);
}

template <typename T>
void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const T, 4>& inGrad, AllocatedTensor<device::CUDA, T>* outGrad[8]) {
    const int length = inGrad.shape().numel();
    ::kernels::decomposeOutputGrad<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, inGrad[0].getDevice().stream()>>>(
        inGrad[0].getDataPtr(), inGrad[1].getDataPtr(), inGrad[2].getDataPtr(), inGrad[3].getDataPtr(),
        outGrad[0]->getDataPtr(), outGrad[1]->getDataPtr(), outGrad[2]->getDataPtr(), outGrad[3]->getDataPtr(),
        outGrad[4]->getDataPtr(), outGrad[5]->getDataPtr(), outGrad[6]->getDataPtr(), outGrad[7]->getDataPtr(),
        length);
}

template <typename T>
void recomposeQuaternionOutput(AllocatedTensor<device::CUDA, T>* inLanes[8], TensorSplit<device::CUDA, T, 4>& outQuats) {
    const int length = outQuats.shape().numel();
    ::kernels::recomposeOutput<<<ceili(length, NUM_THREADS), NUM_THREADS, 0, outQuats[0].getDevice().stream()>>>(
        inLanes[0]->getDataPtr(), inLanes[1]->getDataPtr(), inLanes[2]->getDataPtr(), inLanes[3]->getDataPtr(),
        inLanes[4]->getDataPtr(), inLanes[5]->getDataPtr(), inLanes[6]->getDataPtr(), inLanes[7]->getDataPtr(),
        outQuats[0].getDataPtr(), outQuats[1].getDataPtr(), outQuats[2].getDataPtr(), outQuats[3].getDataPtr(),
        length);
}

template <typename T>
void recomposeQuaternionInputsGrad(AllocatedTensor<device::CUDA, T>* inLeftGradLanes[8], TensorSplit<device::CUDA, T, 4>& outLeftGradQuats,
                                   AllocatedTensor<device::CUDA, T>* inRightGradLanes[8], TensorSplit<device::CUDA, T, 4>& outRightGradQuats) {
    const int leftLen = outLeftGradQuats.shape().numel();
    ::kernels::recomposeLeftInputGrad<<<ceili(leftLen, NUM_THREADS), NUM_THREADS, 0, outLeftGradQuats[0].getDevice().stream()>>>(
        inLeftGradLanes[0]->getDataPtr(), inLeftGradLanes[1]->getDataPtr(), inLeftGradLanes[2]->getDataPtr(), inLeftGradLanes[3]->getDataPtr(),
        inLeftGradLanes[4]->getDataPtr(), inLeftGradLanes[5]->getDataPtr(), inLeftGradLanes[6]->getDataPtr(), inLeftGradLanes[7]->getDataPtr(),
        outLeftGradQuats[0].getDataPtr(), outLeftGradQuats[1].getDataPtr(), outLeftGradQuats[2].getDataPtr(), outLeftGradQuats[3].getDataPtr(),
        leftLen);
    const int rightLen = outRightGradQuats.shape().numel();
    ::kernels::recomposeRightInputGrad<<<ceili(rightLen, NUM_THREADS), NUM_THREADS, 0, outRightGradQuats[0].getDevice().stream()>>>(
        inRightGradLanes[0]->getDataPtr(), inRightGradLanes[1]->getDataPtr(), inRightGradLanes[2]->getDataPtr(), inRightGradLanes[3]->getDataPtr(),
        inRightGradLanes[4]->getDataPtr(), inRightGradLanes[5]->getDataPtr(), inRightGradLanes[6]->getDataPtr(), inRightGradLanes[7]->getDataPtr(),
        outRightGradQuats[0].getDataPtr(), outRightGradQuats[1].getDataPtr(), outRightGradQuats[2].getDataPtr(), outRightGradQuats[3].getDataPtr(),
        rightLen);
}

namespace upstride {
namespace cudnn {

template <>
void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const float, 4>& inLeft, AllocatedTensor<device::CUDA, float>* outLeft[8],
                               const TensorSplit<device::CUDA, const float, 4>& inRight, AllocatedTensor<device::CUDA, float>* outRight[8]) {
    ::decomposeQuaternionInputs(inLeft, outLeft, inRight, outRight);
}

template <>
void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const float, 4>& inGrad, AllocatedTensor<device::CUDA, float>* outGrad[8]) {
    ::decomposeQuaternionOutputGrad(inGrad, outGrad);
}

template <>
void recomposeQuaternionOutput(AllocatedTensor<device::CUDA, float>* inLanes[8], TensorSplit<device::CUDA, float, 4>& outQuats) {
    ::recomposeQuaternionOutput(inLanes, outQuats);
}

template <>
void recomposeQuaternionInputsGrad(AllocatedTensor<device::CUDA, float>* inLeftGradLanes[8], TensorSplit<device::CUDA, float, 4>& outLeftGradQuats,
                                   AllocatedTensor<device::CUDA, float>* inRightGradLanes[8], TensorSplit<device::CUDA, float, 4>& outRightGradQuats) {
    ::recomposeQuaternionInputsGrad(inLeftGradLanes, outLeftGradQuats, inRightGradLanes, outRightGradQuats);
}

#ifdef UPSTRIDE_ENABLE_FP16
template <>
void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const half, 4>& inLeft, AllocatedTensor<device::CUDA, half>* outLeft[8],
                               const TensorSplit<device::CUDA, const half, 4>& inRight, AllocatedTensor<device::CUDA, half>* outRight[8]) {
    ::decomposeQuaternionInputs(inLeft, outLeft, inRight, outRight);
    cudnn::Context::raiseIfError();
}

template <>
void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const half, 4>& inGrad, AllocatedTensor<device::CUDA, half>* outGrad[8]) {
    ::decomposeQuaternionOutputGrad(inGrad, outGrad);
    cudnn::Context::raiseIfError();
}

template <>
void recomposeQuaternionOutput(AllocatedTensor<device::CUDA, half>* inLanes[8], TensorSplit<device::CUDA, half, 4>& outQuats) {
    ::recomposeQuaternionOutput(inLanes, outQuats);
    cudnn::Context::raiseIfError();
}

template <>
void recomposeQuaternionInputsGrad(AllocatedTensor<device::CUDA, half>* inLeftGradLanes[8], TensorSplit<device::CUDA, half, 4>& outLeftGradQuats,
                                   AllocatedTensor<device::CUDA, half>* inRightGradLanes[8], TensorSplit<device::CUDA, half, 4>& outRightGradQuats) {
    ::recomposeQuaternionInputsGrad(inLeftGradLanes, outLeftGradQuats, inRightGradLanes, outRightGradQuats);
    cudnn::Context::raiseIfError();
}
#endif

}  // namespace cudnn
}  // namespace upstride