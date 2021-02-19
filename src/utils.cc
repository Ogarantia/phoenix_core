#include "utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "conv2d.hpp"

using namespace upstride;

int upstride::computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
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
        case Padding::SAME: {
                outputSize = (inputSize + stride - 1) / stride;
                const int paddingNeeded = std::max(0, (outputSize - 1) * stride + effectiveFilterSize - inputSize);
                // For odd values of total padding, add more padding at the 'right' side of the given dimension.
                paddingBefore = paddingNeeded / 2;
                paddingAfter = paddingNeeded - paddingBefore;
                break;
            }
        default:
            throw std::invalid_argument("Invalid padding");
    }

    return outputSize;
}


int upstride::computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                                  int dilation, int stride,
                                                  int paddingBefore,
                                                  int paddingAfter) {
    // Based on Tensorflow implementation:
    // https://github.com/tensorflow/tensorflow/blob/8f7e34982dde766b3fc73c90bcdbfccc001fe8e3/tensorflow/core/framework/kernel_shape_util.cc#L18-L65

    const int effectiveFilterSize = (filterSize - 1) * dilation + 1;
    return (inputSize + paddingBefore + paddingAfter - effectiveFilterSize + stride) / stride;
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