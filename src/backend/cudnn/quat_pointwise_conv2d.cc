#include "quat_pointwise_conv2d.hpp"
#include "../../conv2d.hpp"


namespace upstride {
namespace cuda {


// 'using' declarations for better code readability and partial template specialization
using ConvManager = QuatKernelPointwiseConvManager<device::CUDA>;

template<typename T>
using ForwardFunctor = QuatKernelPointwiseConvForwardFunctor<device::CUDA, T>;

template<typename T>
using BackwardFunctor = QuatKernelPointwiseConvBackwardFunctor<device::CUDA, T>;


// Generic Manager

ConvManager::QuatKernelPointwiseConvManager(
        const Algebra algebra, const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
): filterLayout(filterLayout)
{
    if (algebra == Algebra::QUATERNION && stride.x == 1 && stride.y == 1 && dataFormat == DataFormat::NCHW &&
        (filterLayout == FilterLayout::OHWI || filterLayout == FilterLayout::OIHW)) {
        // the only place where eligibleToRun is set
        eligibleToRun = true;
    }
}


void ConvManager::configure(
    device::CUDA& device, const Shape& inputShape, const Shape& weightsShape,
    const IntPair& padBefore, const IntPair& padAfter, int groups
) {
    // only run if eligibleToRun was set to true in the constructor
    if (eligibleToRun) {
        const IntPair zeroPad(0);
        if (padBefore != zeroPad || padAfter != zeroPad || groups != 1) {
            // remove the control configs
            pointwiseConv = false;
            cached = false;
            return;
        }

        registersPerThreadBlock = device.getRegistersPerThreadBlock();

        // do not reconfigure if the parameters necessary for convDesc haven't changed
        if (inputShape == this->inputShape && weightsShape == this->weightsShape) {
            return;
        }

        cached = false;

        Conv2DFilterLayout filter(filterLayout, Algebra::QUATERNION);

        // check if the convolution is a pointwise one, by inspecting convolution filter size
        auto weightsHeight = filter.height(weightsShape);
        auto weightsWidth = filter.width(weightsShape);
        auto weightsSize = weightsHeight * weightsWidth;
        pointwiseConv = (weightsSize == 1);

        // set convolution descriptor only for pointwise convolutions
        if (pointwiseConv) {
            // store shapes used to decide if a reconfiguration is needed the next time 'configure' is run
            this->inputShape = inputShape;
            this->weightsShape = weightsShape;

            // fully specify the convolution descriptor
            convDesc.outputChannels = filter.numOutputChannels(weightsShape);
            convDesc.inputChannels = filter.numInputChannels(weightsShape);
            convDesc.imageSize = inputShape.width(DataFormat::NCHW) * inputShape.height(DataFormat::NCHW);
            convDesc.batchSize = inputShape[0] / 4;
        }
    }
}


bool ConvManager::canRun() const {
    return (eligibleToRun && pointwiseConv);
}


void ConvManager::validateRun() const {
    // additional check for canRun, which should always be checked before using the operator()
    if (!canRun()) {
        throw std::logic_error("Functor not eligible to run");
    }

    // check if convDesc is set properly
    if (convDesc.batchSize == 0
        || convDesc.imageSize == 0
        || convDesc.inputChannels == 0
        || convDesc.outputChannels == 0
    ) {
        const std::string error_msg = "Found 0 in convolution descriptor dimensions: " + convDesc.toString();
        throw std::logic_error(error_msg);
    }
}


// Forward Functor

template<typename T>
ForwardFunctor<T>::QuatKernelPointwiseConvForwardFunctor(
        const upstride::Context& context, const Algebra algebra,
        const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation)
    : ConvManager(algebra, dataFormat, filterLayout, stride, dilation),
    ConvKernelProfiler<ForwardKernelPtr, T>(context)
{}


template<typename T>
const std::vector<ConvKernelConfiguration>& ForwardFunctor<T>::getForwardConfigs() {
    static const std::vector<ConvKernelConfiguration> forwardKernelConfigs {
        {ConvKernelType::pointwiseForward_2DSharedMemory, {8, 8, 1}},
        {ConvKernelType::pointwiseForward_2DSharedMemory, {16, 16, 1}},
        {ConvKernelType::pointwiseForward_2DSharedMemory, {32, 32, 1}},
    };

    return forwardKernelConfigs;
}


template<typename T>
bool ForwardFunctor<T>::tryCacheOptimalKernel(
    device::CUDA& device, const TensorsPack& tensors
) {
    bool kernelsProfiledLocally {false};
    if (!this->cached) {
        UPSTRIDE_SAYS("Caching forward functor kernel locally");

        auto findResult = this->findOptimalKernel(device, ConvType::forward, this->convDesc, getForwardConfigs(), tensors);
        forwardOptimalKernel = findResult.first;
        kernelsProfiledLocally = !(findResult.second);
        if (!interpretKernelConf(forwardOptimalKernel.conf, this->convDesc, forwardKernelPack)) {
            // mismatch, the found optimal kernel does not match the convolution descriptor
            throw std::logic_error("Failed to set kernel");
        }

        this->cached = true;
    }
    return kernelsProfiledLocally;
}


template<typename T>
void ForwardFunctor<T>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const T>& inputTensor, const Tensor<device::CUDA, const T>& weightsTensor,
    const Tensor<device::CUDA, const T>* biasTensor, Tensor<device::CUDA, T>& outputTensor
) {
#ifdef UPSTRIDE_DEBUG
    this->validateRun();
#endif
    const TensorsPack tensors {inputTensor, weightsTensor, biasTensor, outputTensor};
    bool kernelsProfiledLocally = tryCacheOptimalKernel(device, tensors);
    if (!kernelsProfiledLocally) {
        launchKernel(forwardKernelPack, this->convDesc, tensors);
    }
}


// Backward Functor

template<typename T>
BackwardFunctor<T>::QuatKernelPointwiseConvBackwardFunctor(
        const upstride::Context& context, const Algebra algebra,
        const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
) : ConvManager(algebra, dataFormat, filterLayout, stride, dilation),
    ConvKernelProfiler<BackwardKernelPtr, T>(context)
{}


template<typename T>
const std::vector<ConvKernelConfiguration>& BackwardFunctor<T>::getInputGradConfigs() {
    static const std::vector<ConvKernelConfiguration> inputGradKernelConfigs {
        {ConvKernelType::pointwiseInputGrad_2DSharedMemory, {8, 8, 1}},
        {ConvKernelType::pointwiseInputGrad_2DSharedMemory, {16, 16, 1}},
        {ConvKernelType::pointwiseInputGrad_2DSharedMemory, {32, 32, 1}},
    };

    return inputGradKernelConfigs;
}


template<typename T>
const std::vector<ConvKernelConfiguration>& BackwardFunctor<T>::getWeightsGradConfigs() {
    static const std::vector<ConvKernelConfiguration> weightsGradKernelConfigs {
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {4, 4, 32}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {8, 8, 8}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {16, 16, 2}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {4, 4, 64}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {8, 8, 16}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {16, 16, 4}},
        {ConvKernelType::pointwiseWeightsGrad_3DThreadBlock, {32, 32, 1}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 16, 1}, {1, 1}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 16, 1}, {1, 2}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 16, 1}, {2, 1}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 16, 1}, {2, 2}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 32, 1}, {1, 1}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 32, 1}, {1, 2}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 32, 1}, {2, 1}},
        {ConvKernelType::pointwiseWeightsGrad_AccumulatorsInRegisters, {32, 32, 1}, {2, 2}},
    };

    return weightsGradKernelConfigs;
}


template<typename T>
void BackwardFunctor<T>::launchKernels(
    const TensorsPack& tensorsInputGrad, const TensorsPack& tensorsWeightsGrad
) {
    launchKernel(inputGradKernelPack, this->convDesc, tensorsInputGrad);
    launchKernel(weightsGradKernelPack, this->convDesc, tensorsWeightsGrad);
}


template<typename T>
bool BackwardFunctor<T>::tryCacheOptimalKernels(
    device::CUDA& device, const TensorsPack& tensorsInputGrad, const TensorsPack& tensorsWeightsGrad
) {
    bool inputGradKernelsProfiledLocally {false};
    bool weightsGradKernelsProfiledLocally {false};
    if (!this->cached) {

        // input gradient kernel caching
        {
            UPSTRIDE_SAYS("Caching backward functor input gradient kernel locally");

            auto findResult = this->findOptimalKernel(device, ConvType::inputGrad, this->convDesc, getInputGradConfigs(), tensorsInputGrad);
            inputGradOptimalKernel = findResult.first;
            inputGradKernelsProfiledLocally = !(findResult.second);
            if (!interpretKernelConf(inputGradOptimalKernel.conf, this->convDesc, inputGradKernelPack)) {
                // mismatch, the found optimal kernel does not match the convolution descriptor
                throw std::logic_error("Failed to set kernel");
            }
        }

        // weights gradient kernel caching
        {
            UPSTRIDE_SAYS("Caching backward functor weights gradient kernel locally");

            auto findResult = this->findOptimalKernel(device, ConvType::weightsGrad, this->convDesc, getWeightsGradConfigs(), tensorsWeightsGrad);
            weightsGradOptimalKernel = findResult.first;
            weightsGradKernelsProfiledLocally = !(findResult.second);
            if (!interpretKernelConf(weightsGradOptimalKernel.conf, this->convDesc, weightsGradKernelPack)) {
                // mismatch, the found optimal kernel does not match the convolution descriptor
                throw std::logic_error("Failed to set kernel");
            }
        }

        this->cached = true;
    }
    return (inputGradKernelsProfiledLocally && weightsGradKernelsProfiledLocally);
}


template<typename T>
void BackwardFunctor<T>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const T>& inputTensor, const Tensor<device::CUDA, const T>& weightsTensor,
    const Tensor<device::CUDA, const T>& gradTensor, Tensor<device::CUDA, T>& weightsGradTensor,
    Tensor<device::CUDA, T>& inputGradTensor
) {
#ifdef UPSTRIDE_DEBUG
    this->validateRun();
#endif
    const TensorsPack tensorsInputGrad {weightsTensor, gradTensor, nullptr, inputGradTensor};
    const TensorsPack tensorsWeightsGrad {inputTensor, gradTensor, nullptr, weightsGradTensor};
    bool kernelsProfiledLocally = tryCacheOptimalKernels(device, tensorsInputGrad, tensorsWeightsGrad);
    if (!kernelsProfiledLocally) {
        launchKernels(tensorsInputGrad, tensorsWeightsGrad);
    }
}


// Forward declarations with specialized templates

template ForwardFunctor<float>::QuatKernelPointwiseConvForwardFunctor(
    const upstride::Context& context, const Algebra algebra,
    const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
);

template BackwardFunctor<float>::QuatKernelPointwiseConvBackwardFunctor(
    const upstride::Context& context, const Algebra algebra,
    const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
);

template void ForwardFunctor<float>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const float>& inputTensor, const Tensor<device::CUDA, const float>& weightsTensor,
    const Tensor<device::CUDA, const float>* biasTensor, Tensor<device::CUDA, float>& outputTensor
);

template void BackwardFunctor<float>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const float>& inputTensor, const Tensor<device::CUDA, const float>& weightsTensor,
    const Tensor<device::CUDA, const float>& gradTensor, Tensor<device::CUDA, float>& weightsGradTensor,
    Tensor<device::CUDA, float>& inputGradTensor
);

#ifdef UPSTRIDE_ENABLE_FP16

template ForwardFunctor<half>::QuatKernelPointwiseConvForwardFunctor(
    const upstride::Context& context, const Algebra algebra,
    const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
);

template BackwardFunctor<half>::QuatKernelPointwiseConvBackwardFunctor(
    const upstride::Context& context, const Algebra algebra,
    const DataFormat dataFormat, const FilterLayout filterLayout, const IntPair& stride, const IntPair& dilation
);

template void ForwardFunctor<half>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const half>& inputTensor, const Tensor<device::CUDA, const half>& weightsTensor,
    const Tensor<device::CUDA, const half>* biasTensor, Tensor<device::CUDA, half>& outputTensor
);

template void BackwardFunctor<half>::operator()(
    device::CUDA& device,
    const Tensor<device::CUDA, const half>& inputTensor, const Tensor<device::CUDA, const half>& weightsTensor,
    const Tensor<device::CUDA, const half>& gradTensor, Tensor<device::CUDA, half>& weightsGradTensor,
    Tensor<device::CUDA, half>& inputGradTensor
);

#endif

}
}