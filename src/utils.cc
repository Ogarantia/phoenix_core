#include "utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "conv2d.hpp"

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
inline int computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                               int dilation, int stride,
                                               Padding padding,
                                               int& paddingBefore,
                                               int& paddingAfter) {
    // Based on Tensorflow implementation:
    // https://github.com/tensorflow/tensorflow/blob/8f7e34982dde766b3fc73c90bcdbfccc001fe8e3/tensorflow/core/framework/kernel_shape_util.cc#L18-L65

    const int effectiveFilterSize = (filterSize - 1) * dilation + 1;
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

bool upstride::getSpatialStep(const IntTuple& tuple, int validBatchAndChannelVal, IntPair& result) {
    switch (tuple.size()) {
        case 1:
            result.x = result.y = tuple[0];
            return true;
        case 2:
            result.x = tuple[0];
            result.y = tuple[1];
            return true;
        case 4:
            result.x = tuple[1];
            result.y = tuple[2];
            return tuple[0] == validBatchAndChannelVal && tuple[3] == validBatchAndChannelVal;
    };
    return false;
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

Shape upstride::computeConvOutputSize(Algebra algebra, const DataFormat dataFormat,
                                      const Shape& inputShape, const Shape& filterShape,
                                      Padding paddingPreset,
                                      const IntTuple& explicitPaddings,
                                      const IntTuple& strides,
                                      const IntTuple& dilations,
                                      IntPair& padBefore, IntPair& padAfter,
                                      int groups) {
    // Perform shape checks
    if (inputShape.getSize() != 4)
        throw std::invalid_argument("Four-dimensional input tensor expected");
    if (algebra != Algebra::REAL) {
        if (filterShape.getSize() != 5)
            throw std::invalid_argument("Five-dimensional filter tensor expected");
        if (filterShape[0] != MULTIVECTOR_DIM[algebra])
            throw std::invalid_argument("First filter dimension mismatch, got " + std::to_string(filterShape[0]));
    } else if (filterShape.getSize() != 4)
        throw std::invalid_argument("Four-dimensional filter tensor expected");

    // Grab strides and dilations, check
    IntPair stride, dilation;
    if (!getSpatialStep(strides, 1, stride))
        throw std::invalid_argument("Invalid stides.");
    if (!getSpatialStep(dilations, 1, dilation))
        throw std::invalid_argument("Invalid dilations.");

    // Set up the resulting shape
    Shape outputShape(4);
    outputShape[0] = inputShape[0];
    outputShape.depth(dataFormat) = filterShape[Conv2DKernelLayout::numOutputChannelsDim(algebra)];

    // init padding
    if (paddingPreset == Padding::EXPLICIT) {
        // fixme: this is pretty much not how explicit padding must be implemented
        if (!getSpatialStep(explicitPaddings, 1, padBefore))
            throw std::invalid_argument("Invalid explicit paddings.");
        padAfter = padBefore;
    }

    // compute output size
    outputShape.height(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.height(dataFormat), filterShape[Conv2DKernelLayout::heightDim(algebra)],
        dilation.x, stride.x, paddingPreset,
        padBefore.x, padAfter.x);

    outputShape.width(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.width(dataFormat), filterShape[Conv2DKernelLayout::widthDim(algebra)],
        dilation.y, stride.y, paddingPreset,
        padBefore.y, padAfter.y);

    return outputShape;
}

Algebra upstride::getAlgebraFromType(int uptype) {
    switch (uptype) {
        case 0:
            return Algebra::REAL;
        case 1:
            return Algebra::COMPLEX;
        case 2:
            return Algebra::QUATERNION;
        case 3:
            return Algebra::GA_300;
    }
    throw std::invalid_argument("Invalid datatype index: " + std::to_string(uptype));
}