/**
 * @file backend.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Backend interface declaration
 * This file
 * (1) defines basic datatypes required for the whole stack and
 * (2) declares operations to be implemented in every backend.
 * To be explicitly included only in the backend level.
 * @copyright Copyright (c) 2020 UpStride.io
 */

#pragma once

#include <mutex>
#include "../algebras.hpp"
#include "tensor.hpp"
#include "types.hpp"

/**
 * @brief Defining UPSTRIDE_SAYS macro used for debugging.
 * To enable the verbose mode of the engine, assign to UPSTRIDE_VERBOSE environment variable a non-empty value, e.g.
 *   UPSTRIDE_VERBOSE=1 python test.py
 * This only works when UPSTRIDE_DEBUG macro is defined in compilation to avoid the debugging format string appear
 * in the compiled binary.
 */
#ifdef UPSTRIDE_DEBUG
#define UPSTRIDE_SAYS(CTX, FMT, ...) (CTX).verbosePrintf("\033[1;33m" FMT "\033[0m\n", ##__VA_ARGS__)
#else
#define UPSTRIDE_SAYS(...)
#endif

namespace upstride {

/**
 * @brief Specifies how convolutions are performed for 16-bit floating points inputs and outputs
 */
enum class ConvFp16ComputePolicy {
    FULL_16,                    //!< always use 16-bit floating point computations
    FORWARD_16_BACKWARD_32,     //!< compute forward pass in 16 bits mode and backward pass in 32 bits mode
    FULL_32                     //!< always use 32-bit floating point computations
};

/**
 * @brief Base class of a context shared between different operations
 */
class Context {
   private:
    const bool envVerbose;
    const bool envOptimizeMemoryUse;
    const ConvFp16ComputePolicy convFp16ComputePolicy;
    uint32_t kernelCounter; //!< kernel reference counter.
    std::mutex mutex;
   protected:
    Context();

    /**
     * @brief Called when the number of kernels references by the current context reach 0.
     */
    virtual void cleanUp() = 0;

   public:
    /**
     * @brief Defines whether the processing speed is preferred to the memory consumption, when an implementation choice is available.
     * @return true when it is allowed to consume more memory for better speed.
     * @return false when it is preferable to use less memory at the cost of slower processing.
     */
    inline bool preferSpeedToMemory() const { return !envOptimizeMemoryUse; }

    inline bool isFp16ConvForwardAllowed() const {
        return convFp16ComputePolicy == ConvFp16ComputePolicy::FULL_16 || convFp16ComputePolicy == ConvFp16ComputePolicy::FORWARD_16_BACKWARD_32;
    }

    inline bool isFp16ConvBackwardAllowed() const {
        return convFp16ComputePolicy == ConvFp16ComputePolicy::FULL_16;
    }

    /**
     * @brief Increase the number of kernels associated to the current context.
     */
    inline void increaseKernelCounter() {
        std::lock_guard<std::mutex> lock(mutex);
        kernelCounter++;
    }

    /**
     * @brief Decrease the number of kernels associated to the current context and clean up memory if needed.
     */
    inline void decreaseKernelCounter() {
        std::lock_guard<std::mutex> lock(mutex);
        kernelCounter--;
        if (kernelCounter == 0) {
            cleanUp();
        }
    }

    void verbosePrintf(const char* format, ...) const;
};

/**
 * @brief Scalar 2D convolution operation
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarConv2DFunctor;

/**
 * @brief Scalar 2D convolution operation gradient
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarConv2DGradFunctor;

/**
 * @brief Scalar dense operation
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarDenseFunctor;

/**
 * @brief Scalar dense operation gradient
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class ScalarDenseGradFunctor;


namespace cuda {

/**
 * @brief Generic manager for NCHW quaternion pointwise convolution kernels
 *
 * @tparam Device       Non-dummy implementation only for device::CUDA
 */
template<typename Device>
class QuatKernelPointwiseConvManager {
public:
    /**
     * @brief Constructor for the manager and the inherited classes
     *
     * @param context                       global context
     * @param algebra                       manager is only used with quaternions
     * @param dataFormat                    manager is only used with NCHW data format
     * @param stride                        convolution stride
     * @param dilation                      convolution dilation
     */
    QuatKernelPointwiseConvManager(
        const upstride::Context& context, const Algebra algebra,
        const DataFormat dataFormat, const IntPair& stride, const IntPair& dilation
    ) {}

    /**
     * @brief Checks if the convolution is pointwise, if so configures the manager
     *
     * @param inputShape                    shape of the input data tensor
     * @param weightsShape                  shape of the weights tensor
     * @param padBefore                     convolution padding before parameter
     * @param padAfter                      convolution padding after parameter
     * @param groups                        convolution groups paramater
     */
    void configure(
        Device& device, const Shape& inputShape, const Shape& weightsShape,
        const IntPair& padBefore, const IntPair& padAfter, int groups
    ) {}

    /**
     * @brief Checks if the manager is properly configured for NCHW CUDA pointwise convolutions
     *
     * Expected to always be run before using the operator() from the derived functor classes
     */
    bool canRun() const {
        return false;
    }
};


/**
 * @brief Forward pass functor for NCHW quaternion pointwise convolution using CUDA kernels
 *
 * @tparam Device       Non-dummy implementation only for device::CUDA
 * @tparam T            A scalar datatype
 */
template<typename Device, typename T>
class QuatKernelPointwiseConvForwardFunctor : public QuatKernelPointwiseConvManager<Device> {
public:
    // inheriting constructors from the manager
    using QuatKernelPointwiseConvManager<Device>::QuatKernelPointwiseConvManager;

    /**
     * @brief Operator used to compute convolution forward pass
     *
     * @param device                        device associated with the tensors, holds global kernel configurations cache
     * @param inputTensor                   input data tensor
     * @param weightsTensor                 weights tensor
     * @param biasTensor                    bias tensor
     * @param outputTensor                  output data tensor
     */
    void operator()(
        Device& device,
        const Tensor<Device, const T>& inputTensor, const Tensor<Device, const T>& weightsTensor,
        const Tensor<Device, const T>* biasTensor, Tensor<Device, T>& outputTensor
    ) {
        throw std::logic_error("Functor not eligible to run");
    }
};


/**
 * @brief Backward pass functor for NCHW quaternion pointwise convolution using CUDA kernels
 *
 * @tparam Device       Non-dummy implementation only for device::CUDA
 * @tparam T            A scalar datatype
 */
template<typename Device, typename T>
class QuatKernelPointwiseConvBackwardFunctor : public QuatKernelPointwiseConvManager<Device> {
public:
    // inheriting constructors from the manager
    using QuatKernelPointwiseConvManager<Device>::QuatKernelPointwiseConvManager;

    /**
     * @brief Operator used to compute convolution backward pass
     *
     * @param device                        device associated with the tensors, holds global kernel configurations cache
     * @param inputTensor                   input data tensor
     * @param weightsTensor                 weights tensor
     * @param gradTensor                    gradient tensor, received from the successive layer
     * @param weightsGradTensor             tensor for gradient with respect to the weights, to be computed
     * @param inputGradTensor               tensor for gradient with respect to the input data, to be computed
     */
    void operator()(
        Device& device,
        const Tensor<Device, const T>& inputTensor, const Tensor<Device, const T>& weightsTensor,
        const Tensor<Device, const T>& gradTensor, Tensor<Device, T>& weightsGradTensor,
        Tensor<Device, T>& inputGradTensor
    ) {
        throw std::logic_error("Functor not eligible to run");
    }
};


} // namespace cuda

}  // namespace upstride