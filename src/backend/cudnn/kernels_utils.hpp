#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cuda_runtime.h>

#include "../tensor.hpp"


namespace upstride {
namespace cuda {

/**
 * @brief String representation of a CUDA struct dim3
 *
 * @param block                             dim3 to be represented
 */
inline std::string dim3ToString(const dim3& block) {
    return "(" + std::to_string(block.x) + ", " + std::to_string(block.y) + ", " + std::to_string(block.z) + ")";
}


/**
 * @brief Types of kernel convolution operations
 */
enum class ConvType {
    invalid = 0,
    forward,
    inputGrad,
    weightsGrad
};


/**
 * @brief Available CUDA kernels for different convolution operations
 */
enum class ConvKernelType {
    invalid = 0,

    // pointwise convolution kernels
    pointwiseForward_2DSharedMemory,

    pointwiseInputGrad_2DSharedMemory,

    pointwiseWeightsGrad_3DThreadBlock,
    pointwiseWeightsGrad_AccumulatorsInRegisters
};


/**
 * @brief Descriptor containing chosen tensor sizes for a convolution
 */
struct ConvDesc {
    int inputChannels;                      //!< number of channels in input data
    int outputChannels;                     //!< number of channles in output data
    int imageSize;                          //!< size (width * height) of the input in pixels
    int batchSize;                          //!< number of images in a batch

    /**
     * @brief Overloaded "==" operator to compare two ConvDesc's
     *
     * @param another                       ConvDesc to compare with
     * @return                              true if both ConvDesc's are equal member-wise
     */
    inline bool operator==(const ConvDesc& another) const {
        return (this->inputChannels == another.inputChannels)
            && (this->outputChannels == another.outputChannels)
            && (this->imageSize == another.imageSize)
            && (this->batchSize == another.batchSize);
    }

    /**
     * @brief String representation of the descriptor
     *
     * @return                              string representation of the descriptor
     */
    std::string toString() const;
};


/**
 * @brief Configuration of a convolution operation kernel
 *
 * Contains all the parameters needed for full template specialization of the kernels
 */
struct ConvKernelConfiguration {
    ConvKernelType kernel;                  //!< CUDA kernel type
    dim3 threads;                           //!< CUDA thread block dimensions
    std::pair<int, int> config;             //!< special configuration parameters, currently used only for pointwiseWeightsGrad_AccumulatorsInRegisters

    /**
     * @brief String representation of the configuration
     * Consists of the kernel name, thread blocks size and additional configuration parameters if applicable
     *
     * @return                              string representation of the configuration
     */
    std::string toString() const {
        std::string confStr {};

        static const std::map<ConvKernelType, std::string> kernelTypesToNames {
            { ConvKernelType::pointwiseForward_2DSharedMemory, "pointwiseForward_2DSharedMemory" },
            { ConvKernelType::pointwiseInputGrad_2DSharedMemory, "pointwiseInputGrad_2DSharedMemory" },
            { ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, "pointwiseWeightsGrad_3DThreadBlock" },
            { ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, "pointwiseWeightsGrad_AccumulatorsInRegisters" },
        };

        if (kernelTypesToNames.find(kernel) == kernelTypesToNames.end()) {
            throw std::invalid_argument("Invalid kernel chosen");
        }

        confStr = kernelTypesToNames.at(kernel);
        confStr = confStr + ", thread block: " + dim3ToString(threads);

        // additional config parameters are only applicable to pointwiseWeightsGrad_AccumulatorsInRegisters
        if (kernel == ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters) {
            confStr = confStr + ", conf: ("
                + std::to_string(config.first) + ", "
                + std::to_string(config.second) + ")";
        }

        return confStr;
    }
};


/**
 * @brief Performance profiling statistics
 */
struct PerfRecord {
    float runningTimeAverage;               //!< CUDA kernel execution time in ms, averaged over a couple of rounds
};


/**
 * @brief Combination of a CUDA kernel configuration and its profiling statistics
 */
struct PerfResult {
    ConvKernelConfiguration conf;           //!< CUDA kernel configuration
    PerfRecord perf;                        //!< performance profiling record
};


/**
 * @brief Packaged CUDA kernel, ready to be run
 *
 * @tparam KernelPtr                        type of the CUDA kernel function pointer
 */
template<typename KernelPtr>
struct ConvKernelPack {
    KernelPtr kernel;                       //!< pointer to a CUDA kernel
    dim3 blocks;                            //!< CUDA grid dimensions
    dim3 threads;                           //!< CUDA thread block dimensions
};


/**
 * @brief Packaged tensors on which kernel convolution operations are run
 *
 * @tparam Device                           type of the device the tensors belong to
 * @tparam T                                scalar datatype
 */
template<typename Device, typename T>
struct ConvKernelTensorsPack {
    const Tensor<Device, const T>& inputTensor1;            //!< first input data tensor
    const Tensor<Device, const T>& inputTensor2;            //!< second input data tensor
    const Tensor<Device, const T>* inputTensor3;            //!< third input data tensor (used only for bias)
    Tensor<Device, T>& outputTensor;                        //!< output data tensor
};


/**
 * @brief Cache keeper for CUDA kernel configurations and their profiling results for differconfigurationsent convolution configurations
 */
class ConvKernelsCache {
public:
    /**
     * @brief Check cache for the convolution configuration, set the corresponding result as optimalConf if found
     *
     * @param convType                      kernel convolution operation
     * @param convDesc                      descriptor of the convolution tensor sizes
     * @param optimalConf                   if a configuration is found in cache, the corresponding kernel configuration and profiling record are passed in it
     * @return                              true if the convolution descriptor is found in the cache
     */
    bool checkCache(const ConvType convType, const ConvDesc& convDesc, PerfResult& optimalConf);

    /**
     * @brief Add a profiling result to cache
     *
     * @param convType                      kernel convolution operation
     * @param convDesc                      descriptor of the convolution tensor sizes
     * @param optimalConf                   profiling result to be stored
     */
    void addToCache(const ConvType convType, const ConvDesc& convDesc, const PerfResult& optimalConf);

private:
    /**
     * @brief Type of the
     */
    using Cache = typename std::vector<std::pair<ConvDesc, PerfResult>>;    // could be potentially replaced with a map

    std::map<ConvType, Cache> cachedConfigurations {};      //!< map of cached configurations for different kernel convolution operations
    std::mutex accessControl;                               //!< used to ensure exclusive access to the class
};


}
}