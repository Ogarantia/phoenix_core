#include "debug_utils.hpp"
#include <cuda.h>

using namespace upstride;

static std::mutex accessControl;
static onednn::Context debuggingContext;
static device::CPU deviceCPU(debuggingContext);
static device::CPU allocatorCPU(debuggingContext);

/**
 * @brief Copy tensor from GPU to host
 *
 * @param tensorHost  tensor on CPU
 * @param tensor      tensor on GPU
 * @param device      CUDA device
 */
template <typename T>
void copyTensorFromGpuToHost(TemporaryTensor<device::CPU, const T>& tensorHost,
                             const Tensor<device::CUDA, const T>& tensor,
                             device::CUDA& device) {
  cudaMemcpyAsync(const_cast<T*>(tensorHost.getDataPtr()), tensor.getDataPtr(), tensor.getShape().numel()*sizeof(T), cudaMemcpyDeviceToHost, device.stream());
}

template <typename T>
void copyTensorFromGpuToHost(TemporaryTensor<device::CPU, T>& tensorHost,
                             Tensor<device::CUDA, T>& tensor,
                             device::CUDA& device) {
  cudaMemcpyAsync(tensorHost.getDataPtr(), tensor.getDataPtr(), tensor.getShape().numel()*sizeof(T), cudaMemcpyDeviceToHost, device.stream());
}

/**
 * @brief Get the maximum absolute difference with a point-to-point comparison
 *
 * @param lhs   left array
 * @param rhs   right array
 * @param size  size of arrays
 * @return T
 */
template <typename T>
T getMaxDiff(const T* lhs, const T* rhs, size_t size) {
    T diff = 0;
    for (size_t i = 0; i < size; ++i) {
        diff = std::max(std::abs(lhs[i] - rhs[i]), diff);
    }
    return diff;
}

// TODO Improve position considering an ND localisation and not as a linear array
/**
 * @brief Print the elements whose absolute difference is greater than 0.01 and their position.
 *
 * @param lhs   left array
 * @param rhs   right array
 * @param size  size of arrays
 */
template <typename T>
void highlightDifferences(const T* lhs, const T* rhs, size_t size) {
    T diff = 0;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(lhs[i] - rhs[i]) > 1E-2)
            std::cout << "[Diff] pos["<< i <<" / "<<size<<"] lhs = " << lhs[i] << " | rhs = " << rhs[i] << " | abs(lhs - rhs) = " << std::abs(lhs[i] - rhs[i]) << std::endl;
    }
}

/**
 * @brief Debug function that compares GPU/CPU outputs for Conv2D the forward pass.
 *
 * @tparam
 * @param device        A device instance
 * @param inputTensor   Input tensor
 * @param filterTensor  Filter tensor
 * @param biasTensor    Pointer to bias tensor; may be null
 * @param outputTensor  Output tensor previously compute on GPU
 * @param descriptor    Operation descriptor
 */
template <>
void upstride::conv2DFwdTest<device::CUDA, float>(device::CUDA& device,
                                                  const Tensor<device::CUDA, const float>& inputTensor,
                                                  const Tensor<device::CUDA, const float>& filterTensor,
                                                  const Tensor<device::CUDA, const float>* biasTensor,
                                                  Tensor<device::CUDA, float>& outputTensor,
                                                  const Conv2DFwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(accessControl);
    auto& refOp = deviceCPU.template getConv2DFwdOperation<device::CPU, UpstrideConv2DFunctor<device::CPU, float>>(descriptor);
    MemoryRequest mem(allocatorCPU, refOp);

    TemporaryTensor<device::CPU, const float> inputTensorHost (deviceCPU, mem, inputTensor.getShape());
    TemporaryTensor<device::CPU, const float> filterTensorHost(deviceCPU, mem, filterTensor.getShape());
    TemporaryTensor<device::CPU, const float> biasTensorHost  (deviceCPU, mem, biasTensor ? biasTensor->getShape(): Shape());
    TemporaryTensor<device::CPU, float> outputTensorHost(deviceCPU, mem, outputTensor.getShape());
    TemporaryTensor<device::CPU, float> testOutputTensorHost(deviceCPU, mem, outputTensor.getShape());

    mem.submit();

    inputTensorHost.prepare();
    filterTensorHost.prepare();
    biasTensorHost.prepare();
    outputTensorHost.prepare();
    testOutputTensorHost.prepare();

    // Copy all tensors from GPU memory to host
    copyTensorFromGpuToHost(inputTensorHost, inputTensor, device);
    copyTensorFromGpuToHost(filterTensorHost, filterTensor, device);
    if (biasTensor)
      copyTensorFromGpuToHost(biasTensorHost, *biasTensor, device);
    copyTensorFromGpuToHost(testOutputTensorHost, outputTensor, device);

    // Compute the convolution on CPU
    refOp(deviceCPU, inputTensorHost, filterTensorHost, biasTensor ? &biasTensorHost : nullptr, outputTensorHost, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
    // Get the maximum absolute difference
    float diff = getMaxDiff(outputTensorHost.getDataPtr(), testOutputTensorHost.getDataPtr(), outputTensorHost.getShape().numel());
    if (diff > 1E-2)
        highlightDifferences(outputTensorHost.getDataPtr(), testOutputTensorHost.getDataPtr(), outputTensorHost.getShape().numel());
}

/**
 * @brief Debug function that compares GPU/CPU outputs for the Conv2D backward pass
 *
 * @tparam
 * @param device            A device instance
 * @param inputTensor       Input tensor
 * @param filterTensor      Filter tensor
 * @param gradTensor        Gradient tensor
 * @param filterGradTensor  Filter gradient tensor
 * @param inputGradTensor   Input gradient tensor
 * @param descriptor        Operation descriptor
 */
template <>
void upstride::conv2DBwdTest<device::CUDA, float>(device::CUDA& device,
                                                  const Tensor<device::CUDA, const float>& inputTensor,
                                                  const Tensor<device::CUDA, const float>& filterTensor,
                                                  const Tensor<device::CUDA, const float>& gradTensor,
                                                  Tensor<device::CUDA, float>& filterGradTensor,
                                                  Tensor<device::CUDA, float>& inputGradTensor,
                                                  const Conv2DBwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(accessControl);
    auto& refOp = deviceCPU.template getConv2DBwdOperation<device::CPU, UpstrideConv2DGradFunctor<device::CPU, float>>(descriptor);
    MemoryRequest mem(allocatorCPU, refOp);

    TemporaryTensor<device::CPU, const float> inputTensorHost (deviceCPU, mem, inputTensor.getShape());
    TemporaryTensor<device::CPU, const float> filterTensorHost(deviceCPU, mem, filterTensor.getShape());
    TemporaryTensor<device::CPU, const float> gradTensorHost  (deviceCPU, mem, gradTensor.getShape());
    TemporaryTensor<device::CPU, float> filterGradTensorHost(deviceCPU, mem, filterGradTensor.getShape());
    TemporaryTensor<device::CPU, float> inputGradTensorHost(deviceCPU, mem, inputGradTensor.getShape());
    TemporaryTensor<device::CPU, float> testFilterTensorHost(deviceCPU, mem, filterGradTensor.getShape());
    TemporaryTensor<device::CPU, float> testInputTensorHost(deviceCPU, mem, inputGradTensor.getShape());

    mem.submit();

    inputTensorHost.prepare();
    filterTensorHost.prepare();
    gradTensorHost.prepare();
    filterGradTensorHost.prepare();
    inputGradTensorHost.prepare();
    testFilterTensorHost.prepare();
    testInputTensorHost.prepare();

    // Copy all tensors from GPU memory to host
    copyTensorFromGpuToHost(inputTensorHost, inputTensor, device);
    copyTensorFromGpuToHost(filterTensorHost, filterTensor, device);
    copyTensorFromGpuToHost(gradTensorHost, gradTensor, device);
    copyTensorFromGpuToHost(testFilterTensorHost, filterGradTensor, device);
    copyTensorFromGpuToHost(testInputTensorHost, inputGradTensor, device);

    // Compute the convolution on CPU
    refOp(deviceCPU, inputTensorHost, filterTensorHost, gradTensorHost, filterGradTensorHost, inputGradTensorHost, descriptor.getPaddingBefore(), descriptor.getPaddingAfter(), descriptor.getGroups());
    // Get the maximum absolute difference of FilterGrad
    float diffFilter = getMaxDiff(filterGradTensorHost.getDataPtr(), testFilterTensorHost.getDataPtr(), filterGradTensorHost.getShape().numel());
    if (diffFilter > 1E-2) {
        std::cout << "[FilterGrad]" << std::endl;
        highlightDifferences(filterGradTensorHost.getDataPtr(), testFilterTensorHost.getDataPtr(), filterGradTensorHost.getShape().numel());
    }
    // Get the maximum absolute difference of InputGrad
    float diffInput = descriptor.isInputGradientRequired() ? getMaxDiff(inputGradTensorHost.getDataPtr(), testInputTensorHost.getDataPtr(), inputGradTensorHost.getShape().numel()) : 0.0f;
    if (descriptor.isInputGradientRequired() && diffInput > 1E-2) {
        std::cout << "[InputGrad]" << std::endl;
        highlightDifferences(inputGradTensorHost.getDataPtr(), testInputTensorHost.getDataPtr(), inputGradTensorHost.getShape().numel());
    }
}

/**
 * @brief Debug function that compares GPU/CPU outputs for Dense the forward pass.
 *
 * @tparam
 * @param device        A device instance
 * @param inputTensor   Input tensor
 * @param filterTensor  Filter tensor
 * @param biasTensor    Pointer to bias tensor; may be null
 * @param outputTensor  Output tensor previously compute on GPU
 * @param descriptor    Operation descriptor
 */
template <>
void upstride::denseFwdTest<device::CUDA, float>(device::CUDA& device,
                                                 const Tensor<device::CUDA, const float>& inputTensor,
                                                 const Tensor<device::CUDA, const float>& filterTensor,
                                                 const Tensor<device::CUDA, const float>* biasTensor,
                                                 Tensor<device::CUDA, float>& outputTensor,
                                                 const DenseFwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(accessControl);
    auto& refOp = deviceCPU.template getDenseFwdOperation<device::CPU, UpstrideDenseFunctor<device::CPU, float>>(descriptor);
    MemoryRequest mem(allocatorCPU, refOp);

    TemporaryTensor<device::CPU, const float> inputTensorHost (deviceCPU, mem, inputTensor.getShape());
    TemporaryTensor<device::CPU, const float> filterTensorHost(deviceCPU, mem, filterTensor.getShape());
    TemporaryTensor<device::CPU, const float> biasTensorHost  (deviceCPU, mem, biasTensor ? biasTensor->getShape(): Shape());
    TemporaryTensor<device::CPU, float> outputTensorHost(deviceCPU, mem, outputTensor.getShape());
    TemporaryTensor<device::CPU, float> testOutputTensorHost(deviceCPU, mem, outputTensor.getShape());

    mem.submit();

    inputTensorHost.prepare();
    filterTensorHost.prepare();
    biasTensorHost.prepare();
    outputTensorHost.prepare();
    testOutputTensorHost.prepare();

    //Copy all tensors from GPU memory to host
    copyTensorFromGpuToHost(inputTensorHost, inputTensor, device);
    copyTensorFromGpuToHost(filterTensorHost, filterTensor, device);
    if (biasTensor)
      copyTensorFromGpuToHost(biasTensorHost, *biasTensor, device);
    copyTensorFromGpuToHost(testOutputTensorHost, outputTensor, device);

    // Compute the convolution on CPU
    refOp(deviceCPU, inputTensorHost, filterTensorHost, biasTensor ? &biasTensorHost : nullptr, outputTensorHost);
    // Get the maximum absolute difference
    float diff = getMaxDiff(outputTensorHost.getDataPtr(), testOutputTensorHost.getDataPtr(), outputTensorHost.getShape().numel());
    if (diff > 1E-2)
        highlightDifferences(outputTensorHost.getDataPtr(), testOutputTensorHost.getDataPtr(), outputTensorHost.getShape().numel());
}

/**
 * @brief Debug function that compares GPU/CPU outputs for the Dense backward pass
 *
 * @tparam
 * @param device            A device instance
 * @param inputTensor       Input tensor
 * @param filterTensor      Filter tensor
 * @param gradTensor        Gradient tensor
 * @param filterGradTensor  Filter gradient tensor
 * @param inputGradTensor   Input gradient tensor
 * @param descriptor        Operation descriptor
 */
template <>
void upstride::denseBwdTest<device::CUDA, float>(device::CUDA& device,
                                                 const Tensor<device::CUDA, const float>& inputTensor,
                                                 const Tensor<device::CUDA, const float>& filterTensor,
                                                 const Tensor<device::CUDA, const float>& gradTensor,
                                                 Tensor<device::CUDA, float>& filterGradTensor,
                                                 Tensor<device::CUDA, float>& inputGradTensor,
                                                 const DenseBwdDescriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(accessControl);
    auto& refOp = deviceCPU.template getDenseBwdOperation<device::CPU, UpstrideDenseGradFunctor<device::CPU, float>>(descriptor);
    MemoryRequest mem(allocatorCPU, refOp);

    TemporaryTensor<device::CPU, const float> inputTensorHost (deviceCPU, mem, inputTensor.getShape());
    TemporaryTensor<device::CPU, const float> filterTensorHost(deviceCPU, mem, filterTensor.getShape());
    TemporaryTensor<device::CPU, const float> gradTensorHost  (deviceCPU, mem, gradTensor.getShape());
    TemporaryTensor<device::CPU, float> filterGradTensorHost(deviceCPU, mem, filterGradTensor.getShape());
    TemporaryTensor<device::CPU, float> inputGradTensorHost(deviceCPU, mem, inputGradTensor.getShape());
    TemporaryTensor<device::CPU, float> testFilterTensorHost(deviceCPU, mem, filterGradTensor.getShape());
    TemporaryTensor<device::CPU, float> testInputTensorHost(deviceCPU, mem, inputGradTensor.getShape());

    mem.submit();

    inputTensorHost.prepare();
    filterTensorHost.prepare();
    gradTensorHost.prepare();
    filterGradTensorHost.prepare();
    inputGradTensorHost.prepare();
    testFilterTensorHost.prepare();
    testInputTensorHost.prepare();

    //Copy all tensors from GPU memory to host
    copyTensorFromGpuToHost(inputTensorHost, inputTensor, device);
    copyTensorFromGpuToHost(filterTensorHost, filterTensor, device);
    copyTensorFromGpuToHost(gradTensorHost, gradTensor, device);
    copyTensorFromGpuToHost(testFilterTensorHost, filterGradTensor, device);
    copyTensorFromGpuToHost(testInputTensorHost, inputGradTensor, device);

    refOp(deviceCPU, inputTensorHost, filterTensorHost, gradTensorHost, filterGradTensorHost, inputGradTensorHost);
    // Get the maximum absolute difference of FilterGrad
    float diffFilter = getMaxDiff(filterGradTensorHost.getDataPtr(), testFilterTensorHost.getDataPtr(), filterGradTensorHost.getShape().numel());
    if (diffFilter > 1E-2) {
        std::cout << "[FilterGrad]" << std::endl;
        highlightDifferences(filterGradTensorHost.getDataPtr(), testFilterTensorHost.getDataPtr(), filterGradTensorHost.getShape().numel());
    }
    // Get the maximum absolute difference of InputGrad
    float diffInput = descriptor.isInputGradientRequired() ? getMaxDiff(inputGradTensorHost.getDataPtr(), testInputTensorHost.getDataPtr(), inputGradTensorHost.getShape().numel()) : 0.0f;
    if (descriptor.isInputGradientRequired() && diffInput > 1E-2) {
        std::cout << "[InputGrad]" << std::endl;
        highlightDifferences(inputGradTensorHost.getDataPtr(), testInputTensorHost.getDataPtr(), inputGradTensorHost.getShape().numel());
    }
}
