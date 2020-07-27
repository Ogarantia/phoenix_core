#include "utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "upstride.hpp"

using namespace upstride;

/**
 * @brief Computes output size along a single dimension of an operation that samples the input with strided/dilated patches.
 * @param inputSize         The input size
 * @param filterSize        The patch size
 * @param dilation          The patch dilation
 * @param stride            The patch stride
 * @param padding           Input padding preset
 * @param paddingBefore     Zero padding at the beginning; in case of explicit padding the value is taken as input, otherwise it is computed
 * @param paddingAfter      Zero padding at the end; in case of explicit padding the value is taken as input, otherwise it is computed
 * @return number of samples resulting from the operation.
 */
int computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                        int dilation, int stride,
                                        Padding padding,
                                        int& paddingBefore,
                                        int& paddingAfter) {
    // Based on Tensorflow implementation:
    // https://github.com/tensorflow/tensorflow/blob/8f7e34982dde766b3fc73c90bcdbfccc001fe8e3/tensorflow/core/framework/kernel_shape_util.cc#L18-L65

    int effectiveFilterSize = (filterSize - 1) * dilation + 1;
    int outputSize;
    switch (padding) {
        case Padding::VALID:
            outputSize = (inputSize - effectiveFilterSize + stride) / stride;
            paddingBefore = paddingAfter = 0;
            break;
        case Padding::EXPLICIT:
            outputSize = (inputSize + paddingBefore + paddingAfter - effectiveFilterSize + stride) / stride;
            break;
        case Padding::SAME:
            outputSize = (inputSize + stride - 1) / stride;
            const int paddingNeeded = std::max(0, (outputSize - 1) * stride + effectiveFilterSize - inputSize);
            // For odd values of total padding, add more padding at the 'right' side of the given dimension.
            paddingBefore = paddingNeeded / 2;
            paddingAfter = paddingNeeded - paddingBefore;
            break;
    }

    return outputSize;
}

Padding upstride::paddingFromString(std::string paddingString) {
    if (paddingString == "SAME")
        return Padding::SAME;
    if (paddingString == "VALID")
        return Padding::VALID;
    if (paddingString == "EXPLICIT")
        return Padding::EXPLICIT;
    throw std::invalid_argument("Invalid padding encountered: " + paddingString);
}

DataFormat upstride::dataFormatFromString(std::string dataFormatString) {
    if (dataFormatString == "NHWC")
        return DataFormat::NHWC;
    if (dataFormatString == "NCHW")
        return DataFormat::NCHW;
    throw std::invalid_argument("Invalid data format encountered: " + dataFormatString);
}

Shape upstride::computeConvOutputSize(const int typeDim, const DataFormat dataFormat, const Shape& inputShape, const Shape& filterShape,
                                      Padding paddingPreset, const std::vector<int32_t>& explicitPadding, const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
    // Perform shape checks
    if (inputShape.getSize() != 4)
        throw std::invalid_argument("Four-dimensional input tensor expected");
    if (typeDim > 1) {
        if (filterShape.getSize() != 5)
            throw std::invalid_argument("Five-dimensional filter tensor expected");
        if (filterShape[0] != typeDim)
            throw std::invalid_argument("First filter dimension mismatch, got " + std::to_string(filterShape[0]));
    }
    else
        if (filterShape.getSize() != 4)
            throw std::invalid_argument("Four-dimensional filter tensor expected");

    // For scalar tensors (typeDim == 1), work with usual 4D filters. Otherwise the filter tensor is 5D.
    //fixme: heavily disabled for testing due to oneDNN specificities
    const int firstFilterDim = typeDim > 1 ? 1 : 0;
    const int filterWidthDim = firstFilterDim +3;
    const int filterHeightDim = firstFilterDim + 2;
    const int filterInChannelDim = firstFilterDim + 1;
    const int filterOutChannelDim = firstFilterDim + 0;

    /*if (inputShape.depth(dataFormat) % filterShape[filterInChannelDim] != 0)
        throw std::invalid_argument("Filter channels number/input channels number mismatch");*/

    // Set up the resulting shape
    Shape outputShape(4);
    outputShape[0] = inputShape[0];
    outputShape.depth(dataFormat) = filterShape[filterOutChannelDim];

    // init padding
    int padLeft, padRight, padTop, padBottom;
    if (paddingPreset == Padding::EXPLICIT) {
        padLeft = padRight = explicitPadding[0];
        padTop = padBottom = explicitPadding[1];
    }

    // compute output size
    //fixme: dilation is not taken into account properly
    outputShape.width(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.width(dataFormat), filterShape[filterWidthDim],
        dilation[0], stride[0], paddingPreset,
        padLeft, padRight);

    outputShape.height(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.height(dataFormat), filterShape[filterHeightDim],
        dilation[1], stride[1], paddingPreset,
        padTop, padBottom);

    return outputShape;
}
