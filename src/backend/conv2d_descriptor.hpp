#pragma once
#include "../algebras.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "../utils.hpp"

namespace upstride {

/**
 * @brief Specifies memory layout for a convolution kernel
 *   OIHW for real algebra
 *   nOIHW for a non-real algebra containing n-dimensional multivectors.
 */
class Conv2DKernelLayout {
   public:
    /**
     * @brief Returns number of dimensions in the kernel tensor for a specific algebra
     */
    static inline int rank(Algebra algebra) {
        return algebra == Algebra::REAL ? 4 : 5;
    }

    /**
     * @brief Returns dimension number containing the number of output channels in the convolution kernel for a specific algebra.
     */
    static inline int numOutputChannelsDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 0 : 1;
    }

    /**
     * @brief Returns dimension number containing the number of input channels in the convolution kernel for a specific algebra.
     */
    static inline int numInputChannelsDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 1 : 2;
    }

    /**
     * @brief Returns dimension number containing the height of the convolution kernel for a specific algebra.
     */
    static inline int heightDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 2 : 3;
    }

    /**
     * @brief Returns dimension number containing the width of the convolution kernel for a specific algebra.
     */
    static inline int widthDim(Algebra algebra) {
        return algebra == Algebra::REAL ? 3 : 4;
    }
};


/**
 * @brief Describes 2D convolution operation parameters
 */
class Conv2DDescriptor {
private:
    const Shape inputShape;             //!< operation input tensor shape
    const Shape filterShape;            //!< operation filter tensor shape
    const IntPair stride;               //!< strides along H and W dimensions in pixels
    const IntPair dilation;             //!< dilations along H and W dimensions in pixels
    const int groups;                   //!< number of groups (for group convolutions)
    const Algebra algebra;              //!< algebra corresponding to UpStride datatype
    const DataFormat dataFormat;        //!< input and output tensors data format (channels-first or channels-last)
    const bool useBias;                 //!< if `true`, the bias addition is enabled
    const bool realValuedInput;         //!< if `true`, the input tensor is real-valued
    IntPair padBefore;                  //!< top-left zero-padding applied to the input along H and W dimensions in pixels
    IntPair padAfter;                   //!< bottom-right zero-padding applied to the input along H and W dimensions in pixels

public:
    Conv2DDescriptor(
        const Shape& inputShape,
        const Shape& filterShape,
        IntPair stride,
        IntPair dilation,
        Padding paddingPreset,
        const IntTuple& explicitPadding,
        int groups,
        Algebra algebra,
        DataFormat dataFormat,
        bool useBias,
        bool realValuedInput = false
    ):
        inputShape(inputShape), filterShape(filterShape),
        stride(stride), dilation(dilation), groups(groups), algebra(algebra), dataFormat(dataFormat), useBias(useBias), realValuedInput(realValuedInput)
    {
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

        // init padding
        if (paddingPreset == Padding::EXPLICIT) {
            padBefore = IntPair(explicitPadding[0], explicitPadding[1]);
            padAfter = IntPair(explicitPadding[2], explicitPadding[3]);
        }

        // compute padding
        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filterShape[Conv2DKernelLayout::heightDim(algebra)],
            dilation.x, stride.x, paddingPreset,
            padBefore.x, padAfter.x);

        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filterShape[Conv2DKernelLayout::widthDim(algebra)],
            dilation.y, stride.y, paddingPreset,
            padBefore.y, padAfter.y);
    }

    /**
     * @brief Computes the output tensor shape
     */
    inline Shape getOutputShape() const {
        // Set up the resulting shape
        Shape outputShape(4);
        outputShape[0] = inputShape[0];
        outputShape.depth(dataFormat) = filterShape[Conv2DKernelLayout::numOutputChannelsDim(algebra)];

        // compute output size
        outputShape.height(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filterShape[Conv2DKernelLayout::heightDim(algebra)],
            dilation.x, stride.x,
            padBefore.x, padAfter.x);

        outputShape.width(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filterShape[Conv2DKernelLayout::widthDim(algebra)],
            dilation.y, stride.y,
            padBefore.y, padAfter.y);

        // in case of real-valued input, the output batch size is N times bigger
        if (realValuedInput)
            outputShape[0] = outputShape[0] * upstride::MULTIVECTOR_DIM[algebra];

        return outputShape;
    }

    inline bool operator==(const Conv2DDescriptor& another) const {
        return inputShape == another.inputShape &&
               filterShape == another.filterShape &&
               stride == another.stride &&
               dilation == another.dilation &&
               groups == another.groups &&
               algebra == another.algebra &&
               dataFormat == another.dataFormat &&
               useBias == another.useBias &&
               realValuedInput == another.realValuedInput &&
               padBefore == another.padBefore &&
               padAfter == another.padAfter;
    }

    inline bool operator<(const Conv2DDescriptor& another) const {
        if (inputShape < another.inputShape) return true;
        if (another.inputShape < inputShape) return false;

        if (filterShape < another.filterShape) return true;
        if (another.filterShape < filterShape) return false;

        if (stride < another.stride) return true;
        if (another.stride < stride) return false;

        if (dilation < another.dilation) return true;
        if (another.dilation < dilation) return false;

        if (groups < another.groups) return true;
        if (another.groups < groups) return false;

        if (algebra < another.algebra) return true;
        if (another.algebra < algebra) return false;

        if (dataFormat < another.dataFormat) return true;
        if (another.dataFormat < dataFormat) return false;

        if (useBias < another.useBias) return true;
        if (another.useBias < useBias) return false;

        if (realValuedInput < another.realValuedInput) return true;
        if (another.realValuedInput < realValuedInput) return false;

        if (padBefore < another.padBefore) return true;
        if (another.padBefore < padBefore) return false;

        if (padAfter < another.padAfter) return true;
        if (another.padAfter < padAfter) return false;
        return false;
    }


    inline const IntPair& getStride() const { return stride; }

    inline const IntPair& getDilation() const { return dilation; }

    inline int getGroups() const { return groups; }

    inline Algebra getAlgebra() const { return algebra; }

    inline DataFormat getDataFormat() const { return dataFormat; }

    inline bool isRealValuedInput() const { return realValuedInput; }

    inline bool isBiasUsed() const { return useBias; }

    inline IntPair getPaddingBefore() const { return padBefore; }

    inline IntPair getPaddingAfter() const { return padAfter; }
};

}