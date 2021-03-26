/**
 * @file tensor.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Tensor and Shape classes
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include "types.hpp"

namespace upstride {

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
    static const Shape EMPTY;

    /**
     * @brief Constructs a zero Shape object
     */
    Shape() : size(0), shape(nullptr) {}

    /**
     * @brief Constructs a Shape from a list of numeric values
     * This allows for the inline Shape instantiation like
     *      Shape shape{1, 2, 3, 15};
     * @tparam numeric the datatype of list entries; must be castable to int
     * @param list the list of sizes of every dimension
     */
    template <typename numeric>
    Shape(std::initializer_list<numeric> list) : size(list.size()), shape(new int[size]) {
        int i = 0;
        for (auto _ : list)
            shape[i++] = static_cast<int>(_);
    }

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
     * @brief return true if any value of the size or size is equal to 0
     */
    inline bool empty() const {
        for (int i=0; i<size; i++){
            if (shape[i] == 0) return true;
        }
        return (size == 0);
    }

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
        for (int i = 0; i < size; i++)
            if (shape[i] != another.shape[i])
                return false;
        return true;
    }

    inline bool operator!=(const Shape& another) const {
        return !(*this == another);
    }

    /**
     * @brief Defines a linear order relationship.
     * Enables using Shapes for logarithmic search.
     * @param another       A Shape instance to compare the current instance to
     * @return true if the current shape precedes the other one in the linear ordering sense, false otherwise.
     */
    inline bool operator<(const Shape& another) const {
        return compare(another) < 0;
    }

    inline int compare(const Shape& another) const {
        if (size < another.size) return -1;
        if (another.size < size) return +1;

        // identical size, lexicographic comparaison
        for (int i = 0; i < size; i++) {
            if (shape[i] < another.shape[i]) return -1;
            if (another.shape[i] < shape[i]) return +1;
        }

        return 0;
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
    int64_t numel() const {
	if (size == 0)
	    return 0;
        int64_t numel = shape[0];
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

    /**
     * @brief Cuts out a slice of the Shape in Pythonic fashion
     *      slice(3, 6) contains elements 3, 4, and 5
     *      slice(2) contains all elements starting from the 2nd to the end
     *      slice(-3) contains last three elements
     *      slice(-3, -1) contains elements n - 3 and n - 2
     * @param from  First element index
     * @param to    Last element index. If omitted, keep the tail.
     * @return a slice containing specified elements.
     */
    Shape slice(int from, int to = 0) const {
        if (from < 0)
            from += size;
        if (to <= 0)
            to += size;
        if (from < to && to <= size)
            return Shape(to - from, shape + from);
        throw std::invalid_argument("Slice is out of range: " + std::to_string(from) + ":" + std::to_string(to));
    }

    /**
     * @brief Returns Shape corresponding to splitting of the current Shape on equal parts along the outermost dimension
     * @param numParts  Number of parts to split onto
     * @return the shape of a single part in the split.
     */
    Shape split(int numParts) const {
        if (shape[0] % numParts != 0)
            throw std::invalid_argument("Cannot split " + std::to_string(shape[0]) + " onto " + std::to_string(numParts) + " parts");
        Shape result(size, shape);
        result.shape[0] /= numParts;
        return result;
    }
};

template <typename Device, typename T>
class Tensor;  //!< forward declaration

template <typename Device, typename T, const int PARTS>
class TensorSplit;  //!< forward declaration

template <typename Device, typename T>
class TemporaryTensor;

template <typename Device, typename T>
class AllocatedTensor;


/**
 * @brief Declares a set of tensor manipulation routines.
 * This structure is to be specialized in every backend to bring up the implementation.
 * These routines are not intended to be used directly in the core code. They are wrapped in operators in Tensor class.
 */
template <typename Device>
struct TensorManipulations {
    /**
     * @brief Assigns a vector contents to the tensor.
     * The vector needs to contain the same number of entries as the tensor.
     * @tparam T        scalar datatype
     * @param tensor    the tensor instance to assign the contents to
     * @param contents  the vector containing the data to be assigned to the tensor
     */
    template <typename T>
    static inline void assignContents(Tensor<Device, T>& tensor, const std::vector<T>& contents);

    /**
     * @brief Copies tensor contents to a vector.
     * @tparam tensor_t     input tensor contents scalar datatype
     * @tparam vector_t     output vector scalar datatype
     * @param tensor        the input tensor to copy the data from
     * @param contents      the output vector to copy the data to
     */
    template <typename tensor_t, typename vector_t>
    static inline void getContents(const Tensor<Device, tensor_t>& tensor, std::vector<vector_t>& contents);

    /**
     * @brief Accumulate a tensor (b) to another tensor (a) by addition: a = a + b
     * @tparam T scalar datatype
     * @param input   the tensor values to be added (b)
     * @param output  the destination tensor (a)
     * @param shape   shape of both tensors
     */
    template <typename T>
    static void accumulateAdd(const Tensor<Device, T>& input, Tensor<Device, T>& output, const Shape& shape);

    /**
     * @brief Accumulate a tensor (b) to another tensor (a) by subtraction: a = a - b
     * @tparam T scalar datatype
     * @param input   the tensor values to be subtracted (b)
     * @param output  the destination tensor (a)
     * @param shape   shape of both tensors
     */
    template <typename T>
    static void accumulateSub(const Tensor<Device, T>&, Tensor<Device, T>& output, const Shape& shape);

    /**
     * @brief Maps quaternion tensors as arguments of a multiplicative binary operation onto 8 scalar lanes the operation can be applied to.
     * @tparam T        scalar datatype
     * @param inLeft    left operand
     * @param outLeft   left operand decomposition
     * @param inRight   right operand
     * @param outRight  right operand decomposition
     */
    template <typename T>
    static inline void decomposeQuaternionInputs(const TensorSplit<Device, const T, 4>& inLeft, TemporaryTensor<Device, T>* outLeft,
                                                 const TensorSplit<Device, const T, 4>& inRight, TemporaryTensor<Device, T>* outRight);

    /**
     * @brief Maps output gradient of a multiplicative binary operation onto 8 scalar lanes the backward pass can be performed onto.
     * @tparam T        scalar datatype
     * @param inGrad    operation output gradient
     * @param outGrad   operation gradient decomposition
     */
    template <typename T>
    static inline void decomposeQuaternionOutputGrad(const TensorSplit<Device, const T, 4>& inGrad, TemporaryTensor<Device, T>* outGrad);

    /**
     * @brief Maps 8 scalar lanes of a decomposed quaternion multiplicative binary operation result to a regular quaternion tensor.
     * @tparam T        scalar datatype
     * @param inLanes           operation result
     * @param outQuats          recomposed quaternion output
     */
    template <typename T>
    static inline void recomposeQuaternionOutput(TemporaryTensor<Device, T>* inLanes, TensorSplit<Device, T, 4>& outQuats);

    /**
     * @brief Maps 8 scalar lanes of a decomposed quaternion multiplicative binary operation gradient to quaternion tensors.
     * @tparam T        scalar datatype
     * @param inLeftGradLanes   left operand gradient decomposed on lanes
     * @param outLeftGradQuats  recomposed left operand gradient
     * @param inRightGradLanes  right operand gradient decomposed on lanes
     * @param outRightGradQuats recomposed left operand gradient
     */
    template <typename T>
    static inline void recomposeQuaternionInputsGrad(TemporaryTensor<Device, T>* inLeftGradLanes, TensorSplit<Device, T, 4>& outLeftGradQuats,
                                                     TemporaryTensor<Device, T>* inRightGradLanes, TensorSplit<Device, T, 4>& outRightGradQuats);
};  // namespace tensor_arithmetics

/**
 * @brief Tensor representation using its shape a tensor array
 *
 * @tparam T Type of the tensor
 */
template <typename Device, typename T>
class Tensor {
    Tensor(const Tensor&) = delete;  // disabling copying constructor

   protected:
    Device& device;
    Shape shape;
    T* tensor;  //!< points to the tensor content in memory

   public:
    /**
    * @brief Construct a new Tensor object from an existing content
    * @param sh     Shape of the tensor
    * @param t      pointer to the content
    */
    Tensor(Device& device, const Shape& sh, T* t) : device(device), shape(sh.getSize(), sh.getShapePtr()), tensor(t) {}

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
        TensorManipulations<Device>::accumulateAdd(another, *this);
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
        TensorManipulations<Device>::accumulateSub(another, *this);
        return *this;
    }


    inline Tensor& operator=(const std::vector<T>& contents) {
        TensorManipulations<Device>::assignContents(*this, contents);
        return *this;
    }


    template <typename vector_t>
    inline operator std::vector<vector_t>() const {
        std::vector<vector_t> contents;
        TensorManipulations<Device>::getContents(*this, contents);
        return contents;
    }


    /**
     * @brief Sets all Tensor elements to zero
     * @return the tensor itself
     */
    inline void zero() {
        TensorManipulations<Device>::zero(*this);
    }

#pragma push
#pragma diag_suppress = conversion_function_not_usable  // disabling NVCC warning about unapplied conversion operator
    /**
     * @brief Conversion to the immutable representation of the same data.
     * @return tensor having const-qualified pointer.
     */
    inline operator Tensor<Device, const T>&() {
        return reinterpret_cast<Tensor<Device, const T>&>(*this);
        // static_cast causes endless recursion
    }
#pragma pop

    const Device& getDevice() const {
        return device;
    }

    Device& getDevice() {
        return device;
    }
};

/**
 * @brief Splits a tensor along its outer dimension in smaller tensors.
 * @tparam T        scalar datatype
 * @tparam PARTS    expected number of components
 */
template <typename Device, typename T, const int PARTS>
class TensorSplit {
   private:
    Tensor<Device, T>* parts[PARTS];
    Shape partShape;

    /**
     * @brief Checks inputs and computes the shape of parts.
     * A routine shared among class constructors.
     * @param inputShape shape of the input tensor
     * @param keepOuterDimension if `true`, the outermost input dimension is kept in the parts.
     */
    void initPartShape(const Shape& inputShape, bool keepOuterDimension) {
        if (keepOuterDimension) {
            if (inputShape[0] % PARTS != 0)
                throw std::invalid_argument("Cannot split " + std::to_string(inputShape[0]) + " entries in outer dimension onto " + std::to_string(PARTS) + " parts");
            partShape = inputShape.split(PARTS);
        } else {
            if (inputShape[0] != PARTS)
                throw std::invalid_argument("Expected a tensor of " + std::to_string(PARTS) + " entries in outer dimension, but got " + std::to_string(inputShape[0]));
            partShape = inputShape.slice(1);
        }
    }

   public:
    /**
     * @brief Construct a new TensorSplit object from a big Tensor by cutting it multiple smaller tensors along its outermost dimension.
     * Throws an exception if the outer dimension is not splittable on PARTS equal parts.
     * @param inputTensor           the big tensor
     * @param keepOuterDimension    if `false`, the outermost input tensor dimension is expected to match the number of parts, and the parts shape will have less dimensions.
     *                              Otherwise, the outer dimension is split onto equal parts and preserved in the components even if singleton.
     *                              Example for PARTS = 4:
     *                              [4, x, y, z] input -> keep outer dimension        -> [1, x, y, z] parts
     *                              [4, x, y, z] input -> do not keep outer dimension -> [x, y, z] parts
     *                              [8, x, y, z] input -> keep outer dimension        -> [2, x, y, z] parts
     *                              [8, x, y, z] input -> do not keep outer dimension -> exception is thrown
     *                              [5, x, y, z] input -> exception is thrown anyway (cannot split on PARTS = 4 parts)
     */
    TensorSplit(const Tensor<Device, T>& inputTensor, bool keepOuterDimension = true) {
        const Shape& inputShape = inputTensor.getShape();
        initPartShape(inputShape, keepOuterDimension);
        const T* ptr = inputTensor.getDataPtr();
        const int step = partShape.numel();
        for (int i = 0; i < PARTS; ++i, ptr += step) {
            parts[i] = new Tensor<Device, T>(const_cast<Device&>(inputTensor.getDevice()), partShape, ptr);
        }
    }

    TensorSplit(Tensor<Device, T>& inputTensor, bool keepOuterDimension = true) {
        const Shape& inputShape = inputTensor.getShape();
        initPartShape(inputShape, keepOuterDimension);
        T* ptr = inputTensor.getDataPtr();
        const int step = partShape.numel();
        for (int i = 0; i < PARTS; ++i, ptr += step) {
            parts[i] = new Tensor<Device, T>(inputTensor.getDevice(), partShape, ptr);
        }
    }

    ~TensorSplit() {
        for (int i = 0; i < PARTS; ++i)
            delete parts[i];
    }

    /**
     * @brief Retrieves a part by its index.
     * @param i the index
     * @return i-th part of the splitting.
     */
    inline const Tensor<Device, T>& operator[](int i) const {
        return *parts[i];
    }

    /**
     * @brief Retrieves a part by its index.
     * @param i the index
     * @return i-th part of the splitting.
     */
    inline Tensor<Device, T>& operator[](int i) {
        return *parts[i];
    }

    /**
     * @brief Retrieves the number of parts in the splitting
     * @return the number of parts.
     */
    constexpr inline int numParts() const { return PARTS; }

    /**
     * @brief Returns a shape of a single part
     * @return the part shape.
     */
    inline const Shape& shape() const { return partShape; }
};

}  // namespace upstride
