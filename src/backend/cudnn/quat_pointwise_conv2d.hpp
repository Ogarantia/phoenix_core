#pragma once

#include "tensor.hpp"
#include "algebras.hpp"
#include "../backend.hpp"
#include "../tensor.hpp"
#include "kernels_utils.hpp"
#include "kernels.hpp"

namespace upstride {
namespace cuda {


/**
 * @brief Generic manager for NCHW quaternion pointwise convolution kernels
 */
template<>
class QuatKernelPointwiseConvManager<device::CUDA> {
public:
    /**
     * @brief Constructor for the manager and the inherited classes
     *
     * @param algebra                       manager is only used with quaternions
     * @param dataFormat                    manager is only used with NCHW data format
     * @param stride                        convolution stride
     * @param dilation                      convolution dilation
     */
    QuatKernelPointwiseConvManager(
        const Algebra algebra, const DataFormat dataFormat, const IntPair& stride, const IntPair& dilation
    );

    // default destructor, copy and assignment functions for both reference and move semantics
    virtual ~QuatKernelPointwiseConvManager() = default;

    QuatKernelPointwiseConvManager(QuatKernelPointwiseConvManager&&) = delete;
    QuatKernelPointwiseConvManager& operator=(QuatKernelPointwiseConvManager&&) = delete;

    QuatKernelPointwiseConvManager(const QuatKernelPointwiseConvManager&) = delete;
    QuatKernelPointwiseConvManager& operator=(const QuatKernelPointwiseConvManager&) = delete;

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
        const Shape& inputShape, const Shape& weightsShape,
        const IntPair& padBefore, const IntPair& padAfter, int groups
    );

    /**
     * @brief Checks if the manager is properly configured for NCHW CUDA pointwise convolutions
     *
     * Expected to always be run before using the operator() from the derived functor classes
     */
    bool canRun() const;

protected:
    /**
     * @brief Assert that the manager is properly configured
     */
    void validateRun() const;


    ConvDesc convDesc {0};                  //!< descriptor of the convolution to be run
    bool cached {false};                    //!< determines if the optimal kernel for the convolution is locally cached

private:
    bool pointwiseConv {false};             //!< determines if the convolution to be run is pointwise
    bool eligibleToRun {false};             //!< determines if the manager can be run for the current convolution

    Shape inputShape {};                    //!< shape of the input data tensor
    Shape weightsShape {};                  //!< shape of the weights tensor
};


// Forward Functor

#define FORWARD_KERNEL_PTR void (*) (const T*, const T*, const T*, T*, const int, const int, const int, const int)

/**
 * @brief Forward pass functor for NCHW quaternion pointwise convolution using CUDA kernels
 *
 * @tparam T                                scalar datatype
 */
template<typename T>
class QuatKernelPointwiseConvForwardFunctor<device::CUDA, T> :
    public QuatKernelPointwiseConvManager<device::CUDA>,
    public ConvKernelProfiler<FORWARD_KERNEL_PTR, T>
{
public:
    /**
     * @brief Type definition for kernels computing convolution
     */
    using ForwardKernelPtr = FORWARD_KERNEL_PTR;

#undef FORWARD_KERNEL_PTR

    /**
     * @brief Constructor of the forward functor
     *
     * @param context                       global context
     * @param algebra                       algebra used to compute the convolution
     * @param dataFormat                    manager is only used with NCHW data format
     * @param stride                        convolution stride
     * @param dilation                      convolution dilation
     */
    QuatKernelPointwiseConvForwardFunctor(
        const upstride::Context& context, const Algebra algebra,
        const DataFormat dataFormat, const IntPair& stride, const IntPair& dilation
    );

    // default destructor, copy and assignment functions for both reference and move semantics
    virtual ~QuatKernelPointwiseConvForwardFunctor() = default;

    QuatKernelPointwiseConvForwardFunctor(QuatKernelPointwiseConvForwardFunctor&&) = delete;
    QuatKernelPointwiseConvForwardFunctor& operator=(QuatKernelPointwiseConvForwardFunctor&&) = delete;

    QuatKernelPointwiseConvForwardFunctor(const QuatKernelPointwiseConvForwardFunctor&) = delete;
    QuatKernelPointwiseConvForwardFunctor& operator=(const QuatKernelPointwiseConvForwardFunctor&) = delete;

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
        device::CUDA& device,
        const Tensor<device::CUDA, const T>& inputTensor, const Tensor<device::CUDA, const T>& weightsTensor,
        const Tensor<device::CUDA, const T>* biasTensor, Tensor<device::CUDA, T>& outputTensor
    );

private:
    /**
     * @brief Specialized ConvKernelTensorsPack template
     */
    using TensorsPack = ConvKernelTensorsPack<device::CUDA, T>;
    /**
     * @brief Specialized ConvKernelPack template
     */
    using KernelPack = ConvKernelPack<ForwardKernelPtr>;

    /**
     * @brief Locally cache an optimal convolution kernel if not cached yet
     *
     * @param device                        device associated with the tensors, holds global kernel configurations cache
     * @param tensors                       tensors to run the convolution on
     * @return                              true if kernels profiled locally
     */
    bool tryCacheOptimalKernel(
        device::CUDA& device, const TensorsPack& tensors
    );

    /**
     * @brief Launch a packaged kernel
     *
     * @param kernelPack                    packaged kernel to be launched
     * @param convDesc                      descriptor of the convolution
     * @param tensors                       tensors to run the convolution operation on
     */
    void launchKernel(
        const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
    ) override;

    /**
     * @brief Produce a ready-to-run packaged kernel from a kernel configuration
     *
     * @param conf                          kernel configuration
     * @param convDesc                      descriptor of the convolution
     * @param kernelPack                    paramater used to pass the packaged kernel if the configuration is compatible with the convolution descriptor
     * @return                              true if the configuration is compatible with the convolution descriptor
     */
    bool interpretKernelConf(
        const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
    ) override;

    /**
     * @brief Get available kernel configurations for the forward convolution
     */
    const std::vector<ConvKernelConfiguration>& getForwardConfigs();


    KernelPack forwardKernelPack {nullptr, {0}, {0}};                       //!< packaged ready to run optimal forward convolution kernel
    PerfResult forwardOptimalKernel {{ConvKernelType::invalid}, {0}};       //!< cached optimal forward convolution kernel configuration
};


// Backward Functor

#define BACKWARD_KERNEL_PTR void (*) (const T*, const T*, T*, const int, const int, const int, const int)

/**
 * @brief Backward pass functor for NCHW quaternion pointwise convolution using CUDA kernels
 *
 * @tparam T                                scalar datatype
 */
template<typename T>
class QuatKernelPointwiseConvBackwardFunctor<device::CUDA, T> :
    public QuatKernelPointwiseConvManager<device::CUDA>,
    public ConvKernelProfiler<BACKWARD_KERNEL_PTR, T>
{
public:
    /**
     * @brief Type definition for kernels computing convolution gradient
     */
    using BackwardKernelPtr = BACKWARD_KERNEL_PTR;

#undef BACKWARD_KERNEL_PTR

    /**
     * @brief Constructor of the backward functor
     *
     * @param context                       global context
     * @param algebra                       algebra used to compute the convolution
     * @param dataFormat                    manager is only used with NCHW data format
     * @param stride                        convolution stride
     * @param dilation                      convolution dilation
     */
    QuatKernelPointwiseConvBackwardFunctor(
        const upstride::Context& context, const Algebra algebra,
        const DataFormat dataFormat, const IntPair& stride, const IntPair& dilation
    );

    // default destructor, copy and assignment functions for both reference and move semantics
    virtual ~QuatKernelPointwiseConvBackwardFunctor() = default;

    QuatKernelPointwiseConvBackwardFunctor(QuatKernelPointwiseConvBackwardFunctor&&) = delete;
    QuatKernelPointwiseConvBackwardFunctor& operator=(QuatKernelPointwiseConvBackwardFunctor&&) = delete;

    QuatKernelPointwiseConvBackwardFunctor(const QuatKernelPointwiseConvBackwardFunctor&) = delete;
    QuatKernelPointwiseConvBackwardFunctor& operator=(const QuatKernelPointwiseConvBackwardFunctor&) = delete;

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
        device::CUDA& device,
        const Tensor<device::CUDA, const T>& inputTensor, const Tensor<device::CUDA, const T>& weightsTensor,
        const Tensor<device::CUDA, const T>& gradTensor, Tensor<device::CUDA, T>& weightsGradTensor,
        Tensor<device::CUDA, T>& inputGradTensor
    );

private:
    /**
     * @brief Specialized ConvKernelTensorsPack template
     */
    using TensorsPack = ConvKernelTensorsPack<device::CUDA, T>;
    /**
     * @brief Specialized ConvKernelPack template
     */
    using KernelPack = ConvKernelPack<BackwardKernelPtr>;

    /**
     * @brief Locally cache optimal gradient kernels if not cached yet
     * @param device                        device associated with the tensors, holds global kernel configurations cache
     * @param tensorsInputGrad              tensors needed to compute the input data gradient
     * @param tensorsWeightsGrad            tensors needed to compute the weights gradient
     * @return                              true if kernels profiled locally
     */
    bool tryCacheOptimalKernels(
        device::CUDA& device, const TensorsPack& tensorsInputGrad, const TensorsPack& tensorsWeightsGrad
    );

    /**
     * @brief Launch the computation of gradients with respect to the input data and to the weights
     *
     * @param tensorsInputGrad              tensors needed to compute the input data gradient
     * @param tensorsWeightsGrad            tensors needed to compute the weights gradient
     */
    void launchKernels(
        const TensorsPack& tensorsInputGrad, const TensorsPack& tensorsWeightsGrad
    );

    /**
     * @brief Launch a packaged kernel
     *
     * @param kernelPack                    packaged kernel to be launched
     * @param convDesc                      descriptor of the convolution
     * @param tensors                       tensors to run the convolution operation on
     */
    void launchKernel(
        const KernelPack& kernelPack, const ConvDesc& convDesc, const TensorsPack& tensors
    ) override;

    /**
     * @brief Produce a ready-to-run packaged kernel from a kernel configuration
     *
     * @param conf                          kernel configuration
     * @param convDesc                      descriptor of the convolution
     * @param kernelPack                    paramater used to pass the packaged kernel if the configuration is compatible with the convolution descriptor
     * @return                              true if the configuration is compatible with the convolution descriptor
     */
    bool interpretKernelConf(
        const ConvKernelConfiguration& conf, const ConvDesc& convDesc, KernelPack& kernelPack
    ) override;

    /**
     * @brief Get available kernel configurations for the input data gradient computation
     */
    const std::vector<ConvKernelConfiguration>& getInputGradConfigs();

    /**
     * @brief Get available kernel configurations for the weights gradient computation
     */
    const std::vector<ConvKernelConfiguration>& getWeightsGradConfigs();


    KernelPack inputGradKernelPack {nullptr, {0}, {0}};                     //!< packaged ready to run optimal kernel computing input data gradient
    PerfResult inputGradOptimalKernel {{ConvKernelType::invalid}, {0}};     //!< cached optimal kernel configuration for computing input data gradient

    KernelPack weightsGradKernelPack {nullptr, {0}, {0}};                   //!< packaged ready to run optimal kernel computing weights gradient
    PerfResult weightsGradOptimalKernel {{ConvKernelType::invalid}, {0}};   //!< cached optimal kernel configuration for computing weights gradient
};


}
}