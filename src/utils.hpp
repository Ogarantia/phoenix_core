#pragma once

#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

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
    IntPair(int val): x(val), y(val) {}
    IntPair(int x, int y): x(x), y(y) {}

    inline IntPair operator+(const IntPair & another) const {
        return IntPair(x + another.x, y + another.y);
    }

    inline bool operator==(const IntPair& another) const {
        return x == another.x && y == another.y;
    }
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
    NHWC   // channel-last
};

inline int getWidthDimensionNumber(const DataFormat& dataFormat) {
    static const int DIM_NUMBERS[] = {3, 2};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

inline int getHeightDimensionNumber(const DataFormat& dataFormat) {
    static const int DIM_NUMBERS[] = {2, 1};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

inline int getDepthDimensionNumber(const DataFormat& dataFormat) {
    static int DIM_NUMBERS[] = {1, 3};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

/**
 * @brief Represents shapes of a tensor
 * 
 */
class Shape {
    uint8_t size;
    int* shape;

   public:
    /**
     * @brief Constructs a zero Shape object
     */
    Shape() : size(0), shape(nullptr) {}

    /**
    * @brief Construct a new Shape object
    * @param s Size of the shape array
    * @param _shape Array of shapes
    */
    Shape(int s, const int* _shape) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = _shape[i];
        }
    }

    /**
     * @brief Construct a new Shape object that creates a s size shape with all dimension to 0
     * @param s 
     */
    Shape(int s) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = 0;
        }
    }

    /**
     * @brief Copy constructor
     * @param another       A shape instance to copy from
     */
    Shape(const Shape& another) : size(another.size) {
        shape = new int[size];
        for (int i = 0; i < size; i++)
            shape[i] = another.shape[i];
    }

    /**
     * @brief Assignment operator
     * @param another       A shape instance to copy from
     */
    Shape& operator=(const Shape& another) {
        if (size != another.size) {
            delete[] shape;
            size = another.size;
            shape = new int[size];
        }
        for (int i = 0; i < size; i++)
            shape[i] = another.shape[i];
        return *this;
    }

    /**
     * @brief Destroy the Shape object
     */
    ~Shape() { delete[] shape; }
    uint8_t getSize() const { return size; }
    const int* getShapePtr() const { return shape; }

    /**
     * @brief Accesses shape dimension size by dimension index
     * @param i     A dimension index
     * @return the size of a corresponding dimension
     */
    int operator[](int i) const { return shape[i]; }
    int& operator[](int i) { return shape[i]; }

    /**
     * @brief Accesses the width dimension in function of a specific data format
     * @param fmt   The data format of the tensor
     * @return the tensor width.
     */
    int& width(const DataFormat& fmt) {
        return shape[getWidthDimensionNumber(fmt)];
    }
    int width(const DataFormat& fmt) const {
        return shape[getWidthDimensionNumber(fmt)];
    }

    /**
     * @brief Accesses the height dimension in function of a specific data format
     * @param fmt   The data format of the tensor
     * @return the tensor height.
     */
    int& height(const DataFormat& fmt) {
        return shape[getHeightDimensionNumber(fmt)];
    }
    int height(const DataFormat& fmt) const {
        return shape[getHeightDimensionNumber(fmt)];
    }

    /**
     * @brief Acessess the depth (channel) dimension in function of a specific data format
     * @param fmt   The data format of the tensor
     * @return the tensor depth.
     */
    int& depth(const DataFormat& fmt) {
        return shape[getDepthDimensionNumber(fmt)];
    }
    int depth(const DataFormat& fmt) const {
        return shape[getDepthDimensionNumber(fmt)];
    }

    /**
     * @return the number of elements in the tensor.
     */
    int numel() const {
        int numel = shape[0];
        for (int i = 1; i < size; i++)
            numel *= shape[i];
        return numel;
    }
};

/**
 * @brief Tensor representation using its shape a tensor array
 * 
 * @tparam T Type of the tensor
 */
template <typename T>
class Tensor {
    const Shape shape;
    T* tensor;

   public:
    /**
    * @brief Construct a new Tensor object
    * 
    * @param sh Shape of the tensor
    * @param t Tensor
    */
    Tensor(const Shape& sh, T* t) : shape(sh.getSize(), sh.getShapePtr()),
                                    tensor(t) {}
    /**
     * @brief Get the pointer to the Tensor object 
     * 
     * @return T* Pointer to tensor
     */
    T* getDataPtr() { return tensor; }
    const T* getDataPtr() const { return tensor; }

    /**
     * @brief Get the Shape object
     * 
     * @return const Shape& 
     */
    const Shape& getShape() const { return shape; }
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
 * @return the output tensor shape.
 */
Shape computeConvOutputSize(const int typeDim, const DataFormat dataFormat,
                            const Shape& inputShape, const Shape& filterShape,
                            Padding paddingPreset,
                            const IntTuple& explicitPaddings,
                            const IntTuple& strides,
                            const IntTuple& dilations,
                            IntPair& padBefore, IntPair& padAfter);

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
