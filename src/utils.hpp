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
 * @brief Computes output size and paddings along a single dimension of an operation that samples the input with strided/dilated patches.
 * @param inputSize         The input size
 * @param filterSize        The patch size
 * @param dilation          The patch dilation
 * @param stride            The patch stride
 * @param padding           Input padding preset
 * @param paddingBefore     Resulting zero-padding at the beginning
 * @param paddingAfter      Resulting zero-padding at the end
 * @return number of samples resulting from the operation.
 */
int computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                        int dilation, int stride,
                                        Padding padding,
                                        int& paddingBefore,
                                        int& paddingAfter);

/**
 * @brief Computes output size along a single dimension of an operation that samples the input with strided/dilated patches for a given zero-padding value.
 * @param inputSize         The input size
 * @param filterSize        The patch size
 * @param dilation          The patch dilation
 * @param stride            The patch stride
 * @param paddingBefore     Zero padding at the beginning
 * @param paddingAfter      Zero padding at the end
 * @return number of samples resulting from the operation.
 */
int computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                        int dilation, int stride,
                                        int paddingBefore,
                                        int paddingAfter);

/**
 * @brief Maps abstract user type numbers ("type 1", "type 2", "type 3") to Algebras
 * Raises an exception if the input type number is out of a valid range.
 * @param uptype    The user type number
 * @return Algebra corresponding to the type number
 */
Algebra getAlgebraFromType(int uptype);

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
