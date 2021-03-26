/**
 * @file kernels.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Bunch of helpful CUDA kernels
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include "../backend.hpp"
#include "../tensor.hpp"
#include "device.hpp"
#include "kernels_utils.hpp"

/**
 * @brief Rounding up integer division
 * @param n nominator
 * @param d denominator
 * @return closest integer greater than n/d
 */
static inline int ceili(int n, int d) {
    return (n + d - 1) / d;
}

namespace upstride {
namespace cudnn {

/**
 * @brief Crops a tensor along W and H dimensions.
 * A part of input tensor shifted from its origin by `offset` samples is copied to the output tensor.
 * The size of the coped area equals to the output tensor size.
 * @tparam T            tensor datatype
 * @param input         input tensor
 * @param output        output tensor
 * @param dataFormat    input and output data format
 * @param offset        (H,W) offset counted from the input image origin specifying the output content origin
 */
template <typename T>
extern void crop(const Tensor<device::CUDA, T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset);

/**
 * @brief Insert a tensor into another bigger tensor with a potential offset along the spatial dimensions H and W.
 * A part of output tensor is filled with the input tensor contents. The filled area is shifted from the input origin by `offset` samples.
 * The size of the coped area equals to the output tensor size.
 * @tparam T            tensor datatype
 * @param input         input tensor
 * @param output        output tensor
 * @param dataFormat    input and output tensors data format
 * @param offset        (H,W) offset counted from the output image origin specifying the origin of the copied content
 */
template <typename T>
extern void insert(const Tensor<device::CUDA, const T>& input, Tensor<device::CUDA, T>& output, DataFormat dataFormat, const IntPair& offset);

/**
 * @brief Adds a bias to a tensor
 *
 * @tparam T the tensor datatype
 * @param tensor        The input and the resulting tensor
 * @param bias          The bias tensor
 * @param dataFormat    input and output tensors data format
 */
template <typename T>
extern void addBias(Tensor<device::CUDA, T>& tensor, const Tensor<device::CUDA, const T>& bias, DataFormat dataFormat);

template <typename T>
extern void accumulateAdd(const device::CUDA& device, T* accumulator, const T* term, int length);

template <typename T>
extern void accumulateSub(const device::CUDA& device, T* accumulator, const T* term, int length);

template <typename T>
extern void decomposeQuaternionInputs(const TensorSplit<device::CUDA, const T, 4>& inLeft, TemporaryTensor<device::CUDA, T>* outLeft,
                                      const TensorSplit<device::CUDA, const T, 4>& inRight, TemporaryTensor<device::CUDA, T>* outRight);

template <typename T>
extern void decomposeQuaternionOutputGrad(const TensorSplit<device::CUDA, const T, 4>& inGrad, TemporaryTensor<device::CUDA, T>* outGrad);

template <typename T>
extern void recomposeQuaternionOutput(TemporaryTensor<device::CUDA, T>* inLanes, TensorSplit<device::CUDA, T, 4>& outQuats);

template <typename T>
extern void recomposeQuaternionInputsGrad(TemporaryTensor<device::CUDA, T>* inLeftGradLanes, TensorSplit<device::CUDA, T, 4>& outLeftGradQuats,
                                          TemporaryTensor<device::CUDA, T>* inRightGradLanes, TensorSplit<device::CUDA, T, 4>& outRightGradQuats);

}  // namespace cudnn


namespace cuda {

/**
 * @brief Framework for generic convolution kernels profiling
 */
template<typename KernelPtr, typename T>
class ConvKernelProfiler {
public:
    /**
     * @brief Constructor for the profiler and the inherited classes
     *
     * @param context                       global context
     */
    ConvKernelProfiler(const upstride::Context& context);

    /**
     * @brief Destructor for the profiler and the inherited classes
     */
    virtual ~ConvKernelProfiler();

    // default copy and assignment functions for both reference and move semantics
    ConvKernelProfiler(ConvKernelProfiler&&) = delete;
    ConvKernelProfiler& operator=(ConvKernelProfiler&&) = delete;

    ConvKernelProfiler(const ConvKernelProfiler&) = delete;
    ConvKernelProfiler& operator=(const ConvKernelProfiler&) = delete;

protected:
    /**
     * @brief Specialized ConvKernelTensorsPack template
     */
    using TensorsPack = ConvKernelTensorsPack<device::CUDA, T>;
    /**
     * @brief Specialized ConvKernelPack template
     */
    using KernelPack = ConvKernelPack<KernelPtr>;

    /**
     * @brief Find optimal convolution kernel, either in global cache or by local profiling
     *
     * @param device                        device associated with the tensors, holds global kernel configurations cache
     * @param convType                      type of the kernel convolution operation
     * @param convDesc                      descriptor of the convolution tensor sizes
     * @param configsToRun                  convolution kernel configurations to be profiled if a suitable record is not found in cache
     * @param tensors                       packaged tensors on which kernel convolution operation will be run
     * @return                              optimal kernel configuration and its profiling record, and whether the kernel was found in global cache
     */
    std::pair<PerfResult, bool> findOptimalKernel(
        device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
        const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
    );

    /**
     * @brief Launch a packaged kernel
     *
     * @param kernelPack                    packaged kernel to be launched
     * @param convDesc                      descriptor of the convolution
     * @param tensors                       tensors to run the convolution operation on
     */
    virtual void launchKernel(
        const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
    ) = 0;

    /**
     * @brief Produce a ready-to-run packaged kernel from a kernel configuration
     *
     * @param conf                          kernel configuration
     * @param convDesc                      descriptor of the convolution
     * @param kernelPack                    paramater used to pass the packaged kernel if the configuration is compatible with the convolution descriptor
     * @return                              true if the configuration is compatible with the convolution descriptor
     */
    virtual bool interpretKernelConf(
        const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
    ) = 0;


    const upstride::Context& context;       //!< global context, used to print debug information

private:
    /**
     * @brief Profile the kernels available for the chosen convolution operation
     *
     * @param convDesc                      the descriptor of the convolution tensor sizes
     * @param configsToRun                  convolution kernel configurations to be profiled
     * @param tensors                       packaged tensors on which kernel convolution operation will be run
     *
     * @return                              profiling statistics for the run configurations
     */
    std::vector<PerfResult> measureKernelsPerf(
        const ConvDesc& convDesc, const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
    );

    /**
     * @brief Profile the kernels available for the chosen convolution and pick the optimal one
     *
     * @param convDesc                      the descriptor of the convolution tensor sizes
     * @param configsToRun                  convolution kernel configurations to be profiled
     * @param tensors                       packaged tensors on which kernel convolution operation will be run
     *
     * @return                              profiling statistics for the optimal kernel
     */
    PerfResult profileOptimalKernel(
        const ConvDesc& convDesc, const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
    );

    /**
     * @brief Profile a single kernel
     *
     * @param kernelPack                    ready to run packaged kernel
     * @param convDesc                      the descriptor of the convolution tensor sizes
     * @param tensors                       packaged tensors on which kernel convolution operation will be profiled
     */
    void profileKernel(
        const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
    );

    /**
     * @brief Collect the gathered profiling data for a single kernel
     *
     * @return                              profiling statistics for the last run kernel
     */
    PerfRecord collectProfiledData();

    /**
     * @brief Asserts that a packaged kernel is fully configured
     *
     * @param kernelPack                    packaged kernel to be verified
     */
    void assertKernelSet(const KernelPack& kernelPack) const;


    constexpr static size_t PERF_ROUNDS = 5;//!< number of profiling rounds for each profiled kernel
    cudaEvent_t starts[PERF_ROUNDS];        //!< used to mark the moment when the profiled kernel is launched
    cudaEvent_t stops[PERF_ROUNDS];         //!< used to mark the moment when the profiled kernel is finished
};

} // namespace cuda

}  // namespace upstride