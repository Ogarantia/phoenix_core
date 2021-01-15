/**
 * @file conv2d_algo_select.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Automatic runtime selection of cuDNN 2D convolution algorithm by speed
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

#include <cudnn.h>
#include <vector>
#include <mutex>
#include "../backend.hpp"
#include "../tensor.hpp"

namespace upstride {
namespace cudnn {

/**
 * @brief Automatic runtime selection of the fastest cuDNN 2D convolution algorithm for a given convolution geometry
 * based on an empirical measurement. The measurements are cached and reused for further queries with the same
 * convolution parameters.
 */
class Conv2DAlgorithmSelector {
private:
    /**
     * @brief Complete description of a 2D convolution geometry (tensor sizes, padding, strides, dilations).
     * Used to identify algorithms choices.
     */
    class Conv2DConfigDescriptor {
    private:
        cudnnDataType_t computeType;    //!< datatype used to perform the computation
        cudnnDataType_t tensorType;     //!< input, kernel and output tensor elements datatype (may differ from computeType)
        Shape inputShape, kernelShape;
        IntPair pad, stride, dilation;
        int groups;
        cudnnMathType_t mathType;       //!< math type for the algorithm, using either regular math or Tensor Cores
    public:
        /**
         * @brief Construct a new Conv 2D descriptor object
         * @param convDesc  cuDNN 2D convolution operation descriptor
         * @param input     cuDNN 2D convolution input tensor descriptor
         * @param kernel    cuDNN 2D convolution kernel tensor descriptor
         */
        Conv2DConfigDescriptor(const cudnnConvolutionDescriptor_t& convDesc, const cudnnTensorDescriptor_t& input, const cudnnFilterDescriptor_t& kernel);

        /**
         * @brief Compares the descriptor to another descriptor.
         * @return true if the two describe the same convolution geometry, false otherwise.
         */
        bool operator==(const Conv2DConfigDescriptor&) const;

        /**
         * @brief Prints the convolution configuration in verbose mode.
         * @param context   A context instance
         */
        void printOut(const upstride::Context& context) const;
    };

    typedef struct {
        size_t scratchpadSize;      //!< size in bytes of a memory buffer needed by the algorithm
        float time;                 //!< execution time in ms
        cudnnMathType_t mathType;   //!< math type for the algorithm, using either regular math or Tensor Cores
    } AlgorithmCharacteristics;

    /**
     * @brief A forward algorithm option
     */
    typedef struct {
        AlgorithmCharacteristics characteristics;
        cudnnConvolutionFwdAlgo_t algorithm;
    } ForwardAlgorithmChoice;

    /**
     * @brief A backward filter algorithm
     */
    typedef struct {
        AlgorithmCharacteristics characteristics;
        cudnnConvolutionBwdFilterAlgo_t algorithm;
    } BackwardFilterAlgorithmChoice;

    /**
     * @brief A backwar data algorithm
     */
    typedef struct {
        AlgorithmCharacteristics characteristics;
        cudnnConvolutionBwdDataAlgo_t algorithm;
    } BackwardDataAlgorithmChoice;

    std::mutex accessControl;   //!< ensures thread safety of the algorithm choice mappings
    std::vector<std::pair<Conv2DConfigDescriptor, ForwardAlgorithmChoice>> forwardAlgorithms;                   //!< tested forward configurations and their selected algorithms
    std::vector<std::pair<Conv2DConfigDescriptor, BackwardFilterAlgorithmChoice>> backwardFilterAlgorithms;     //!< tested backward (filter) configurations and their selected algorithms
    std::vector<std::pair<Conv2DConfigDescriptor, BackwardDataAlgorithmChoice>> backwardDataAlgorithms;         //!< tested backward (data) configurations and their selected algorithms

public:
    inline Conv2DAlgorithmSelector() {}

    /**
     * @brief Selects the fastest forward algorithm applicable for a given convolution parameters set.
     * When queried for the first time for a specific parameters set, performs a speed measurement of applicable
     * algorithms by means of cuDNN, which may take a while. Further queries use cached results.
     * @param context           A context instance
     * @param handle            A cuDNN handle
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param output            The convolution output tensor descriptor
     * @param executionTime     Returns execution time in milliseconds took by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    cudnnConvolutionFwdAlgo_t selectForwardAlgo(const upstride::Context& context,
                                                const cudnnHandle_t handle,
                                                const cudnnConvolutionDescriptor_t& convDesc,
                                                const cudnnTensorDescriptor_t& input,
                                                const cudnnFilterDescriptor_t& kernel,
                                                const cudnnTensorDescriptor_t& output,
                                                float& executionTime,
                                                size_t& scratchpadSize,
                                                cudnnMathType_t& mathType);

    /**
     * @brief Selects the fastest backward algorithm computing the filter gradient, applicable for a given convolution
     * parameters set.
     * When queried for the first time for a specific parameters set, performs a speed measurement of applicable
     * algorithms by means of cuDNN, which may take a while. Further queries use cached results.
     * @param context           A context instance
     * @param handle            A cuDNN handle
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param executionTime     Returns execution time in milliseconds took by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    cudnnConvolutionBwdFilterAlgo_t selectBackwardFilterAlgo(const upstride::Context& context,
                                                             const cudnnHandle_t handle,
                                                             const cudnnConvolutionDescriptor_t& convDesc,
                                                             const cudnnTensorDescriptor_t& input,
                                                             const cudnnTensorDescriptor_t& grad,
                                                             const cudnnFilterDescriptor_t& kernel,
                                                             float& executionTime,
                                                             size_t& scratchpadSize,
                                                             cudnnMathType_t& mathType);

    /**
     * @brief Selects the fastest backward algorithm computing the input (data) gradient, applicable for a
     * given convolution parameters set.
     * When queried for the first time for a specific parameters set, performs a speed measurement of applicable
     * algorithms by means of cuDNN, which may take a while. Further queries use cached results.
     * @param context           A context instance
     * @param handle            A cuDNN handle
     * @param convDesc          The 2D convolution operation descriptor
     * @param input             The convolution input tensor descriptor
     * @param grad              The loss function gradient tensor descriptor
     * @param kernel            The convolution kernel (filter) tensor descriptor
     * @param executionTime     Returns execution time in milliseconds took by the selected algorithm
     * @param scratchpadSize    Returns the memory buffer size in bytes needed for the selected algorithm to run
     * @param mathType          Returns math type for the algorithm, either regular math or Tensor Cores
     * @return the fastest algorithm for the given 2D convolution parameter set.
     */
    cudnnConvolutionBwdDataAlgo_t selectBackwardDataAlgo(const upstride::Context& context,
                                                         const cudnnHandle_t handle,
                                                         const cudnnConvolutionDescriptor_t& convDesc,
                                                         const cudnnTensorDescriptor_t& input,
                                                         const cudnnTensorDescriptor_t& grad,
                                                         const cudnnFilterDescriptor_t& kernel,
                                                         float& executionTime,
                                                         size_t& scratchpadSize,
                                                         cudnnMathType_t& mathType);
};

}
}