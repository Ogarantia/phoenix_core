/**
 * @file tensor.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Tensor and Shape classes
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <stdexcept>

namespace upstride {

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
 * @brief A tensor shape
 */
class Shape {
    int size;
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
     * @brief Overloaded "==" operator to compare two Shapes.
     * 
     * @param another Shape
     * @return true if both Shape are equals; size and dimensions
     * @return false 
     */
    inline bool operator==(const Shape& another) const {
        if (size != another.size)
            return false;
        for (size_t i = 0; i < size; i++)
            if (shape[i] != another.shape[i])
                return false;
        return true;
    }

    inline bool operator!=(const Shape& another) const {
        return !(*this == another);
    }

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

    /**
     * @brief Add a dimension of 1 at the "dim" index
     *
     * @param dim    Index where the new dimension must be added
     * @return Shape Return a new Shape with the new dimension.
     */
    Shape expandDim(int dim = 0) const {
        Shape shape(size + 1);
        for (int i = 0; i < size + 1; i++) {
            if (i < dim)
                shape[i] = this->shape[i];
            else if (i == dim)
                shape[i] = 1;
            else
                shape[i] = this->shape[i - 1];
        }
        return shape;
    }
};

template <typename Device, typename T>
class Tensor;  //!< forward declaration

/**
 * @brief Declares a set of tensor manipulation routines.
 * This structure is to be specialized in every backend to bring up the implementation.
 * These routines are not intended to be used directly in the core code. They are wrapped in operators in Tensor class.
 */
template <typename Device>
struct TensorManipulations {
    /**
     * @brief Accumulate a tensor (b) to another tensor (a) by addition: a = a + b
     * @tparam T scalar datatype
     * @param input   the tensor values to be added (b)
     * @param output  the destination tensor (a)
     * @param shape   shape of both tensors
     */
    template <typename T>
    void accumulateAdd(const Tensor<Device, T>& input, Tensor<Device, T>& output, const Shape& shape);

    /**
     * @brief Accumulate a tensor (b) to another tensor (a) by subtraction: a = a - b
     * @tparam T scalar datatype
     * @param input   the tensor values to be subtracted (b)
     * @param output  the destination tensor (a)
     * @param shape   shape of both tensors
     */
    template <typename T>
    void accumulateSub(const Tensor<Device, T>&, Tensor<Device, T>& output, const Shape& shape);

};  // namespace tensor_arithmetics

/**
 * @brief Tensor representation using its shape a tensor array
 * 
 * @tparam T Type of the tensor
 */
template <typename Device, typename T>
class Tensor {
    const Shape shape;

   protected:
    T* tensor;  //!< points to the tensor content in memory

   public:
    /**
    * @brief Construct a new Tensor object from an existing content
    * @param sh     Shape of the tensor
    * @param t      pointer to the content
    */
    Tensor(const Shape& sh, T* t) : shape(sh.getSize(), sh.getShapePtr()),
                                    tensor(t) {}

    /**
     * @brief Get the pointer to the tensor content in memory
     * @return T* Pointer to tensor
     */
    T* getDataPtr() { return tensor; }
    const T* getDataPtr() const { return tensor; }

    /**
     * @brief Retrieves the shape of the Tensor instance
     * @return const Shape& 
     */
    const Shape& getShape() const { return shape; }

    /**
     * @brief Adds a Tensor to the current Tensor
     * @param another   the tensor to add
     * @return the tensor itself
     */
    inline Tensor& operator+=(const Tensor& another) {
        if (shape != another.shape)
            throw std::invalid_argument("Tensor shapes mismatch in accumulate-add");
        TensorManipulations<Device>::accumulateAdd(another, *this, shape);
        return *this;
    }

    /**
     * @brief Subtracts a Tensor to the current Tensor
     * @param another   the tensor to subtract
     * @return the tensor itself
     */
    inline Tensor& operator-=(const Tensor& another) {
        if (shape != another.shape)
            throw std::invalid_argument("Tensor shapes mismatch in accumulate-subtract");
        TensorManipulations<Device>::accumulateSub(another, *this, shape);
        return *this;
    }
};

/**
 * @brief Forward declaration of a Tensor implementation that allocates and frees the memory itself.
 * The implementation is device-dependend and is provided by every backend.
 */
template <typename Device, typename T>
class AllocatedTensor;

}  // namespace upstride