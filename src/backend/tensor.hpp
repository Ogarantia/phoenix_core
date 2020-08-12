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

}  // namespace upstride