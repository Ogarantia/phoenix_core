#pragma once

#include <iterator>
#include <ostream>
#include <string>

#include "backend/api.h"

namespace upstride {

/**
 * @brief Padding preset specification
 */
enum class Padding {
    SAME,
    VALID,
    EXPLICIT
};

/**
 * @brief Retrieves padding preset value from a string.
 * Raises an exception if unable to interpret the string.
 * @param paddingString     The string
 * @return corresponding padding value.
 */
Padding paddingFromString(std::string paddingString);

/**
 * @brief Retrieves data format value from a string.
 * Raises an exception if unable to interpret the string.
 * @param dataFormatString     The string
 * @return corresponding data format value.
 */
DataFormat dataFormatFromString(std::string dataFormatString);

/**
 * @brief Retrieves a spatial step information (stride, dilation) along width and height from a tuple.
 * @param tuple                         The tuple to look at
 * @param validBatchAndChannelVal       A valid value expected for channel and batch dimensions
 * @param result                        The resulting spatial step
 * @return true if the tuple is successfully interpreted.
 * @return false otherwise.
 */
bool getSpatialStep(const IntTuple& tuple, int validBatchAndChannelVal, IntPair& result);

/**
 * @brief Computes convolution output shape
 * The filter memory layout is assumed as follows: [blade, filter height, filter_width, input channels, output channels]
 * @param typeDim           Dimensionality of a specific UpStride datatype (e.g., 4 for quaternions)
 * @param dataFormat        Input and output tensors data format
 * @param inputShape        Input tensor shape
 * @param filterShape       Kernel tensor shape
 * @param paddingPreset     Padding preset
 * @param padding           Explicit padding value if the padding preset is explicit
 * @param strides           Convolution stride
 * @param dilations         Convolution dilation rate
 * @param padBefore         Number of zero samples to add at the beginning to height and width input dimensions (computed in function of other parameters)
 * @param padAfter          Number of zero samples to add at the end to height and width input dimensions (computed in function of other parameters)
 * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
 * @return the output tensor shape.
 */
Shape computeConvOutputSize(const int typeDim, const DataFormat dataFormat,
                            const Shape& inputShape, const Shape& filterShape,
                            Padding paddingPreset,
                            const IntTuple& explicitPaddings,
                            const IntTuple& strides,
                            const IntTuple& dilations,
                            IntPair& padBefore, IntPair& padAfter,
                            int groups = 1);

}  // namespace upstride

namespace std {
/**
 * @brief Overloaded "<<" operator to write out std::vectors to an std::stream. A very handy thing.
 * @tparam T    vector datatype
 * @param str   The output stream
 * @param vec   The vector to write out
 * @return a reference to the output stream, by convention.
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& str, const std::vector<T>& vec) {
    if (!vec.empty()) {
        str << '[';
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(str, ", "));
        str << "\b\b]";
    } else
        str << "[]";
    return str;
}

/**
 * @brief Overloaded "<<" to write out an upstride::shape to an std::stream. A very handy thing.
 * @param str       The output stream
 * @param shape     The instance of upstride::Shape to write out
 * @return a reference to the output stream, by convention.
 */
inline std::ostream& operator<<(std::ostream& str, const upstride::Shape& shape) {
    if (shape.getSize() > 0) {
        str << '[';
        for (int i = 0; i < shape.getSize(); ++i)
            str << shape[i] << ", ";
        str << "\b\b]";
    } else
        str << "[]";

    return str;
}

}  // namespace std

/**
 * @brief Overloaded "==" operator to compare two Shape a and b.
 * 
 * @param a Shape
 * @param b Shape
 * @return true if both Shape are equals; size and dimensions
 * @return false 
 */
inline bool operator==(const upstride::Shape& a, const upstride::Shape& b) {
    if (!(a.getSize() == b.getSize())) return false;
    for (size_t i = 0; i < a.getSize(); i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}
