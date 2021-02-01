#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace upstride {

/**
 * @brief A fairly generic integer tuple
 */
typedef std::vector<int32_t> IntTuple;

/**
 * @brief A lightweight pair of integer numbers
 */
class IntPair {
   public:
    int x, y;

    IntPair() : x(0), y(0) {}
    IntPair(int val) : x(val), y(val) {}
    IntPair(int x, int y) : x(x), y(y) {}

    /**
     * @brief Construct an IntPair from a tuple.
     * If the tuple contains a single element, it is assigned to the both elements of the tuple.
     * If there are two elements, they are taken as is. Otherwise, an exception is thrown.
     * @param tuple         The input tuple
     */
    IntPair(const IntTuple& tuple) {
        if (tuple.size() == 1)
            x = y = tuple[0];
        else if (tuple.size() == 2) {
            x = tuple[0];
            y = tuple[1];
        }
        else
            throw std::invalid_argument("Cannot construct an integer pair from a tuple of " + std::to_string(tuple.size()) + " elements");
    }

    inline IntPair operator+(const IntPair& another) const {
        return IntPair(x + another.x, y + another.y);
    }

    inline IntPair operator-(const IntPair& another) const {
        return IntPair(x - another.x, y - another.y);
    }

    inline IntPair operator*(const IntPair& another) const {
        return IntPair(x * another.x, y * another.y);
    }

    inline IntPair operator/(const IntPair& another) const {
        return IntPair(x / another.x, y / another.y);
    }

    inline bool operator==(const IntPair& another) const {
        return x == another.x && y == another.y;
    }

    inline bool operator!=(const IntPair& another) const {
        return !(*this == another);
    }

    inline bool operator<(const IntPair& another) const {
        return (x < another.x || (x ==  another.x && y < another.y));
    }

    static const IntPair ZEROS;
    static const IntPair ONES;
};

/**
 * @brief Padding preset specification
 */
enum class Padding {
    SAME,
    VALID,
    EXPLICIT
};

/**
 * @brief Data format specification
 */
enum class DataFormat {
    NCHW,  // channel-first
    NHWC,  // channel-last
    NC,    // plain 2D tensor
    CN     // permuted 2D tensor
};

/**
 * @brief Retrieves padding preset value from a string.
 * Raises an exception if unable to interpret the string.
 * @param paddingString     The string
 * @return corresponding padding value.
 */
inline Padding paddingFromString(const std::string& paddingString)  {
    if (paddingString == "SAME")
        return Padding::SAME;
    if (paddingString == "VALID")
        return Padding::VALID;
    if (paddingString == "EXPLICIT")
        return Padding::EXPLICIT;
    throw std::invalid_argument("Invalid padding encountered: " + paddingString);
}

/**
 * @brief Retrieves data format value from a string.
 * Raises an exception if unable to interpret the string.
 * @param dataFormatString     The string
 * @return corresponding data format value.
 */
inline DataFormat dataFormatFromString(const std::string& dataFormatString) {
    if (dataFormatString == "NHWC")
        return DataFormat::NHWC;
    if (dataFormatString == "NCHW")
        return DataFormat::NCHW;
    throw std::invalid_argument("Invalid data format encountered: " + dataFormatString);
}

inline const char* dataFormatToString(const DataFormat format) {
    switch (format) {
        case DataFormat::NCHW:
            return "NCHW";
        case DataFormat::NHWC:
            return "NHWC";
        case DataFormat::NC:
            return "NC";
        case DataFormat::CN:
            return "CN";
    }
    return "";
}

}