#include "conv2d_algo_select.hpp"
#include "context.hpp"

using namespace upstride;


cudnn::Conv2DAlgorithmSelector::Conv2DConfigDescriptor::Conv2DConfigDescriptor(
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& input,
    const cudnnTensorFormat_t inputFormat,
    const cudnnFilterDescriptor_t& kernel
):
    tensorFormat(inputFormat), inputShape(4), kernelShape(4)
{
    // extract convolution geometry description from cuDNN descriptors
    cudnnConvolutionMode_t mode;
    Context::raiseIfError(cudnnGetConvolution2dDescriptor(convDesc, &pad.x, &pad.y, &stride.x, &stride.y,
                                                          &dilation.x, &dilation.y, &mode, &computeType));
    Context::raiseIfError(cudnnGetConvolutionGroupCount(convDesc, &groups));
    Context::raiseIfError(cudnnGetConvolutionMathType(convDesc, &mathType));

    int whatever;   // the value is not used
    Context::raiseIfError(cudnnGetTensor4dDescriptor(
        input, &tensorType, &inputShape[0], &inputShape[1], &inputShape[2], &inputShape[3],
        &whatever, &whatever, &whatever, &whatever));

    cudnnTensorFormat_t format;
    cudnnDataType_t kernelType;
    Context::raiseIfError(cudnnGetFilter4dDescriptor(
        kernel, &kernelType, &format, &kernelShape[0], &kernelShape[1], &kernelShape[2], &kernelShape[3]));
    if (kernelType != tensorType)
        throw std::runtime_error("Input tensor vs kernel tensor datatype mismatch");
}


bool cudnn::Conv2DAlgorithmSelector::Conv2DConfigDescriptor::operator==(const Conv2DConfigDescriptor& other) const {
    return
        inputShape == other.inputShape &&
        kernelShape == other.kernelShape &&
        computeType == other.computeType &&
        pad == other.pad &&
        stride == other.stride &&
        dilation == other.dilation &&
        tensorFormat == other.tensorFormat &&
        mathType == other.mathType;
}


void cudnn::Conv2DAlgorithmSelector::Conv2DConfigDescriptor::printOut() const {
    UPSTRIDE_SAYS("  %d/%d types, %d I/O format, %d*%d*%d*%d x %d*%d*%d*%d, %d*%d strides, %d*%d pad, %d*%d dilation",
                  computeType, tensorType, tensorFormat,
                  inputShape[0], inputShape[1], inputShape[2], inputShape[3],
                  kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3],
                  stride.x, stride.y, pad.x, pad.y, dilation.x, dilation.y);
}


cudnnConvolutionFwdAlgo_t cudnn::Conv2DAlgorithmSelector::selectForwardAlgo(const cudnnHandle_t handle,
                                                                            const cudnnConvolutionDescriptor_t& convDesc,
                                                                            const cudnnTensorDescriptor_t& input,
                                                                            const cudnnFilterDescriptor_t& kernel,
                                                                            const cudnnTensorDescriptor_t& output,
                                                                            const cudnnTensorFormat_t tensorFormat,
                                                                            float& executionTime,
                                                                            size_t& scratchpadSize,
                                                                            cudnnMathType_t& mathType) {
    std::lock_guard<std::mutex> lock(accessControl);

    // build the descriptor
    const Conv2DConfigDescriptor desc(convDesc, input, tensorFormat, kernel);

    // check if a cached result is available
    for (const auto& entry : forwardAlgorithms)
        if (entry.first == desc) {
            UPSTRIDE_SAYS("Reusing a forward conv2D algorithm (%u entries in cache)", forwardAlgorithms.size());
            scratchpadSize = entry.second.characteristics.scratchpadSize;
            executionTime = entry.second.characteristics.time;
            mathType = entry.second.characteristics.mathType;
            return entry.second.algorithm;
        }

    // perform the measurement
    UPSTRIDE_SAYS("Selecting a forward conv2D algorithm");
    desc.printOut();
    static const int MAX_ALGO_COUNT = 16;
    cudnnConvolutionFwdAlgoPerf_t algo[MAX_ALGO_COUNT];
    int algoCount;
    cudnn::Context::raiseIfError(cudnnFindConvolutionForwardAlgorithm(
        handle,
        input, kernel, convDesc, output,
        MAX_ALGO_COUNT, &algoCount, algo
    ));

    // ensure at least one algorithm is available
    cudnn::Context::raiseIfError(algo[0].status);

    // print results (in verbose mode only)
    for (int i = 0; i < algoCount; ++i)
        if (algo[i].status == CUDNN_STATUS_SUCCESS)
            UPSTRIDE_SAYS("   %d type %d: %0.3f ms", algo[i].algo, algo[i].mathType, algo[i].time);

    // pick the best (fastest) option and store it in the cache
    scratchpadSize = algo[0].memory;
    executionTime = algo[0].time;
    mathType = algo[0].mathType;

    forwardAlgorithms.emplace_back(desc, ForwardAlgorithmChoice{ {scratchpadSize, executionTime, mathType}, algo[0].algo });
    return algo[0].algo;
}


cudnnConvolutionBwdFilterAlgo_t cudnn::Conv2DAlgorithmSelector::selectBackwardFilterAlgo(const cudnnHandle_t handle,
                                                                                         const cudnnConvolutionDescriptor_t& convDesc,
                                                                                         const cudnnTensorDescriptor_t& input,
                                                                                         const cudnnTensorDescriptor_t& grad,
                                                                                         const cudnnFilterDescriptor_t& kernel,
                                                                                         const cudnnTensorFormat_t tensorFormat,
                                                                                         float& executionTime,
                                                                                         size_t& scratchpadSize,
                                                                                         cudnnMathType_t& mathType) {
    std::lock_guard<std::mutex> lock(accessControl);

    // build the descriptor
    Conv2DConfigDescriptor desc(convDesc, input, tensorFormat, kernel);

    // check if a cached result is available
    for (const auto& entry : backwardFilterAlgorithms)
        if (entry.first == desc) {
            UPSTRIDE_SAYS("Reusing a backward filter conv2D algorithm (%u entries in cache)", backwardFilterAlgorithms.size());
            scratchpadSize = entry.second.characteristics.scratchpadSize;
            executionTime = entry.second.characteristics.time;
            mathType = entry.second.characteristics.mathType;
            return entry.second.algorithm;
        }

    // perform the measurement
    UPSTRIDE_SAYS("Selecting a backward filter conv2D algorithm");
    desc.printOut();
    static const int MAX_ALGO_COUNT = 16;
    cudnnConvolutionBwdFilterAlgoPerf_t algo[MAX_ALGO_COUNT];
    int algoCount;
    cudnn::Context::raiseIfError(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle,
        input, grad, convDesc, kernel,
        MAX_ALGO_COUNT, &algoCount, algo
    ));

    // ensure at least one algorithm is available
    cudnn::Context::raiseIfError(algo[0].status);

    // print results (in verbose mode only)
    for (int i = 0; i < algoCount; ++i)
        if (algo[i].status == CUDNN_STATUS_SUCCESS)
            UPSTRIDE_SAYS("   %d type %d: %0.3f ms", algo[i].algo, algo[i].mathType, algo[i].time);

    // pick the best (fastest) option and store it in the cache
    scratchpadSize = algo[0].memory;
    executionTime = algo[0].time;
    mathType = algo[0].mathType;

    backwardFilterAlgorithms.emplace_back(desc, BackwardFilterAlgorithmChoice{ {scratchpadSize, executionTime, mathType}, algo[0].algo });
    return algo[0].algo;
}


cudnnConvolutionBwdDataAlgo_t cudnn::Conv2DAlgorithmSelector::selectBackwardDataAlgo(const cudnnHandle_t handle,
                                                                                     const cudnnConvolutionDescriptor_t& convDesc,
                                                                                     const cudnnTensorDescriptor_t& input,
                                                                                     const cudnnTensorDescriptor_t& grad,
                                                                                     const cudnnFilterDescriptor_t& kernel,
                                                                                     const cudnnTensorFormat_t tensorFormat,
                                                                                     float& executionTime,
                                                                                     size_t& scratchpadSize,
                                                                                     cudnnMathType_t& mathType) {
    std::lock_guard<std::mutex> lock(accessControl);

    // build the descriptor
    Conv2DConfigDescriptor desc(convDesc, input, tensorFormat, kernel);

    // check if a cached result is available
    for (const auto& entry : backwardDataAlgorithms)
        if (entry.first == desc) {
            UPSTRIDE_SAYS("Reusing a backward data conv2D algorithm (%u entries in cache)", backwardDataAlgorithms.size());
            scratchpadSize = entry.second.characteristics.scratchpadSize;
            executionTime = entry.second.characteristics.time;
            mathType = entry.second.characteristics.mathType;
            return entry.second.algorithm;
        }

    // perform the measurement
    UPSTRIDE_SAYS("Selecting a backward data conv2D algorithm");
    desc.printOut();
    static const int MAX_ALGO_COUNT = 16;
    cudnnConvolutionBwdDataAlgoPerf_t algo[MAX_ALGO_COUNT];
    int algoCount;
    cudnn::Context::raiseIfError(cudnnFindConvolutionBackwardDataAlgorithm(
        handle,
        kernel, grad, convDesc, input,
        MAX_ALGO_COUNT, &algoCount, algo
    ));

    // ensure at least one algorithm is available
    cudnn::Context::raiseIfError(algo[0].status);

    // print results (in verbose mode only)
    for (int i = 0; i < algoCount; ++i)
        if (algo[i].status == CUDNN_STATUS_SUCCESS)
            UPSTRIDE_SAYS("   %d type %d: %0.3f ms", algo[i].algo, algo[i].mathType, algo[i].time);

    // pick the best (fastest) option and store it in the cache
    scratchpadSize = algo[0].memory;
    executionTime = algo[0].time;
    mathType = algo[0].mathType;

    backwardDataAlgorithms.emplace_back(desc, BackwardDataAlgorithmChoice{ {scratchpadSize, executionTime, mathType}, algo[0].algo });
    return algo[0].algo;
}