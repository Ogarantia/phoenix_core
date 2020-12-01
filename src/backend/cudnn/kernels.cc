#include <algorithm>
#include "kernels.hpp"


namespace upstride {
namespace cuda {

// Convolution Kernels Profiler

template<typename KernelPtr, typename T>
ConvKernelProfiler<KernelPtr, T>::ConvKernelProfiler(const upstride::Context& context)
    : context(context)
{
    // create CUDA events needed for kernels profiling
    for (int i = 0; i < PERF_ROUNDS; i++) {
        cudaEventCreate(&(starts[i]));
        cudaEventCreate(&(stops[i]));
    }
}


template<typename KernelPtr, typename T>
ConvKernelProfiler<KernelPtr, T>::~ConvKernelProfiler() {
    // destroy CUDA events
    for (int i = 0; i < PERF_ROUNDS; i++) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
}


template<typename KernelPtr, typename T>
void ConvKernelProfiler<KernelPtr, T>::assertKernelSet(const KernelPack& kernelPack) const {
    // check if kernel pointer is set properly
    if (kernelPack.kernel == nullptr) {
        throw std::logic_error("Kernel pointer not set");
    }

    // check if thread block dimensions are set properly
    if (kernelPack.threads.x == 0 || kernelPack.threads.y == 0 || kernelPack.threads.z == 0) {
        const std::string error_msg = "Found 0 in thread block dimensions: " + dim3ToString(kernelPack.threads);
        throw std::logic_error(error_msg);
    }

    // check if grid dimensions are set properly
    if (kernelPack.blocks.x == 0 || kernelPack.blocks.y == 0 || kernelPack.blocks.z == 0) {
        const std::string error_msg = "Found 0 in grid dimensions: " + dim3ToString(kernelPack.blocks);
        throw std::logic_error(error_msg);
    }
}


template<typename KernelPtr, typename T>
void ConvKernelProfiler<KernelPtr, T>::profileKernel(
    const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
) {
    const auto& stream = tensors.outputTensor.getDevice().stream();

    // profile a kernel run perfRounds times
    for (int i = 0; i < PERF_ROUNDS; i++) {
        cudaEventRecord(starts[i], stream);
        launchKernel(kernelPack, convDesc, tensors);
        cudaEventRecord(stops[i], stream);
    }

    // wait for the events registered in the stream to be completed
    cuStreamSynchronize(stream);
}


template<typename KernelPtr, typename T>
PerfRecord ConvKernelProfiler<KernelPtr, T>::collectProfiledData() {
    float runningTimeTotal {0};

    // calculate the running time for each profiling round
    for (int i = 0; i < PERF_ROUNDS; i++) {
        float elapsedTime {0};
        cudaEventElapsedTime(&elapsedTime, starts[i], stops[i]);
        runningTimeTotal += elapsedTime;
    }

    // compute the profiling statistics
    float runningTimeAverage = runningTimeTotal / PERF_ROUNDS;
    return PerfRecord{runningTimeAverage};
}


template<typename KernelPtr, typename T>
std::vector<PerfResult> ConvKernelProfiler<KernelPtr, T>::measureKernelsPerf(
    const ConvDesc& convDesc, const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
) {
    std::vector<PerfResult> perfResults;

    // try to profile each available configuration
    for (const auto &conf : configsToRun) {

        // convert a kernel configuration into a ready to run packaged kernel
        KernelPack kernelPack;
        if (!interpretKernelConf(conf, convDesc, kernelPack)) {
            // skip profiling if the kernel configuration is not compatible with the convolution descriptor
            continue;
        }
#ifdef UPSTRIDE_DEBUG
        assertKernelSet(kernelPack);
#endif

        // profile the kernel
        profileKernel(kernelPack, convDesc, tensors);
        PerfRecord kernelPerfRecord = collectProfiledData();

        // store the profiling result
        perfResults.push_back({conf, kernelPerfRecord});
    }
    return perfResults;
}


template<typename KernelPtr, typename T>
PerfResult ConvKernelProfiler<KernelPtr, T>::profileOptimalKernel(
    const ConvDesc& convDesc, const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
) {
    // gather profiling results for all available kernels
    auto perfResults = this->measureKernelsPerf(convDesc, configsToRun, tensors);

    // print profiling results as debug info
    for (const auto& result : perfResults) {
        UPSTRIDE_SAYS(context, "  %0.3f ms : %s",
            result.perf.runningTimeAverage,
            result.conf.toString().c_str()
        );
    }

    // pick the fastest kernel based on the profiling results
    auto fastest = std::min_element(
        std::begin(perfResults), std::end(perfResults),
        [] (const PerfResult &r1, const PerfResult &r2) {
            return r1.perf.runningTimeAverage < r2.perf.runningTimeAverage;
        }
    );

    return *fastest;
}


template<typename KernelPtr, typename T>
std::pair<PerfResult, bool> ConvKernelProfiler<KernelPtr, T>::findOptimalKernel(
    device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
    const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
) {
    PerfResult optimalResult {{ConvKernelType::invalid}, {0}};

    // check if a suitable record is already in the global cache
    bool cachedGlobally = device.checkCacheForOptimalKernel(convType, convDesc, optimalResult);
    if (!cachedGlobally) {
        UPSTRIDE_SAYS(context, "Described convolution not yet registered in cache");
        // profile kernels locally
        optimalResult = profileOptimalKernel(convDesc, configsToRun, tensors);

        // add the profiling result to the global cache
        device.cacheOptimalKernel(convType, convDesc, optimalResult);
    } else {
        UPSTRIDE_SAYS(context, "Reusing kernel from global cache");
    }

    return {optimalResult, cachedGlobally};
}


// utility 'using' declarations for Profiler template specialization
template<typename T>
using ForwardKernelPtr = void (*) (const T*, const T*, const T*, T*, const int, const int, const int, const int);

template<typename T>
using BackwardKernelPtr = void (*) (const T*, const T*, T*, const int, const int, const int, const int);


// Forward declarations with specialized templates

template ConvKernelProfiler<ForwardKernelPtr<float>, float>::ConvKernelProfiler(const upstride::Context& context);
template ConvKernelProfiler<BackwardKernelPtr<float>, float>::ConvKernelProfiler(const upstride::Context& context);

template ConvKernelProfiler<ForwardKernelPtr<float>, float>::~ConvKernelProfiler();
template ConvKernelProfiler<BackwardKernelPtr<float>, float>::~ConvKernelProfiler();

template std::pair<PerfResult, bool> ConvKernelProfiler<ForwardKernelPtr<float>, float>::findOptimalKernel(
    device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
    const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
);

template std::pair<PerfResult, bool> ConvKernelProfiler<BackwardKernelPtr<float>, float>::findOptimalKernel(
    device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
    const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
);

#ifdef UPSTRIDE_ENABLE_FP16

template ConvKernelProfiler<ForwardKernelPtr<half>, half>::ConvKernelProfiler(const upstride::Context& context);
template ConvKernelProfiler<BackwardKernelPtr<half>, half>::ConvKernelProfiler(const upstride::Context& context);

template ConvKernelProfiler<ForwardKernelPtr<half>, half>::~ConvKernelProfiler();
template ConvKernelProfiler<BackwardKernelPtr<half>, half>::~ConvKernelProfiler();

template std::pair<PerfResult, bool> ConvKernelProfiler<ForwardKernelPtr<half>, half>::findOptimalKernel(
    device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
    const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
);

template std::pair<PerfResult, bool> ConvKernelProfiler<BackwardKernelPtr<half>, half>::findOptimalKernel(
    device::CUDA& device, ConvType convType, const ConvDesc& convDesc,
    const std::vector<ConvKernelConfiguration>& configsToRun, const TensorsPack& tensors
);

#endif

}
}