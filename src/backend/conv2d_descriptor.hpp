#pragma once
#include <sstream>
#include "../algebras.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "../utils.hpp"

namespace upstride {

/**
 * @brief Helper class manipulating with memory layout of convolution kernels.
 */
class Conv2DFilterLayout {
    int o, i, h, w;
public:
    inline Conv2DFilterLayout(FilterLayout layout, Algebra algebra = Algebra::REAL) {
        switch (layout) {
        case FilterLayout::OIHW:
            o = 0;
            i = 1;
            h = 2;
            w = 3;
            break;
        case FilterLayout::OHWI:
            o = 0;
            h = 1;
            w = 2;
            i = 3;
            break;
        case FilterLayout::HWIO:
            h = 0;
            w = 1;
            i = 2;
            o = 3;
            break;
        default:
            throw std::invalid_argument("Unsupported Conv2D filter layout");
        }
        // for non-real algebras the outermost kernel dimension is reserved for the multivector dimension
        if (algebra != Algebra::REAL) {
            o++; i++; h++; w++;
        }
    }

    /**
     * @brief Returns number of dimensions in the kernel tensor for a specific algebra
     */
    static inline int rank(Algebra algebra) {
        return algebra == Algebra::REAL ? 4 : 5;
    }

    /**
     * @brief Returns dimension number containing the number of output channels in the convolution kernel.
     */
    inline int numOutputChannelsDim() const {
        return o;
    }

    /**
     * @brief Returns dimension number containing the number of input channels in the convolution kernel.
     */
    inline int numInputChannelsDim() const {
        return i;
    }

    /**
     * @brief Returns dimension number containing the height of the convolution kernel.
     */
    inline int heightDim() const {
        return h;
    }

    /**
     * @brief Returns dimension number containing the width of the convolution kernel.
     */
    inline int widthDim() const {
        return w;
    }

    /**
     * @brief Returns number of output channels in a convolution kernel of a given shape.
     */
    inline int numOutputChannels(const Shape& shape) const {
        return shape[o];
    }

    /**
     * @brief Returns number of input channels in a convolution kernel of a given shape.
     */
    inline int numInputChannels(const Shape& shape) const {
        return shape[i];
    }

    /**
     * @brief Returns height of a convolution kernel of a given shape.
     */
    inline int height(const Shape& shape) const {
        return shape[h];
    }

    /**
     * @brief Returns width of a convolution kernel of a given shape.
     */
    inline int width(const Shape& shape) const {
        return shape[w];
    }
};


/**
 * @brief Describes 2D convolution operation parameters
 */
class Conv2DDescriptor {
protected:
    const Shape inputShape;             //!< operation input tensor shape
    const Shape filterShape;            //!< operation filter tensor shape
    const IntPair stride;               //!< strides along H and W dimensions in pixels
    const IntPair dilation;             //!< dilations along H and W dimensions in pixels
    const int groups;                   //!< number of groups (for group convolutions)
    const Algebra algebra;              //!< algebra corresponding to UpStride datatype
    const DataFormat dataFormat;        //!< input and output tensors data format (channels-first or channels-last)
    const FilterLayout filterLayout;    //!< filter tensor layout spec
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
        FilterLayout filterLayout,
        bool realValuedInput = false
    ):
        inputShape(inputShape), filterShape(filterShape),
        stride(stride), dilation(dilation), groups(groups), algebra(algebra), dataFormat(dataFormat), filterLayout(filterLayout),
        realValuedInput(realValuedInput)
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
        Conv2DFilterLayout filter(filterLayout, algebra);

        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filter.height(filterShape),
            dilation.x, stride.x, paddingPreset,
            padBefore.x, padAfter.x);

        upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filter.width(filterShape),
            dilation.y, stride.y, paddingPreset,
            padBefore.y, padAfter.y);
    }

    /**
     * @brief Computes the output tensor shape
     */
    inline Shape getOutputShape() const {
        // Set up the resulting shape
        Conv2DFilterLayout filter(filterLayout, algebra);
        Shape outputShape(4);
        outputShape[0] = inputShape[0];
        outputShape.depth(dataFormat) = filter.numOutputChannels(filterShape);

        // compute output size
        outputShape.height(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.height(dataFormat), filter.height(filterShape),
            dilation.x, stride.x,
            padBefore.x, padAfter.x);

        outputShape.width(dataFormat) = upstride::computeWindowedOutputSizeAndPadding(
            inputShape.width(dataFormat), filter.width(filterShape),
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
               realValuedInput == another.realValuedInput &&
               padBefore == another.padBefore &&
               padAfter == another.padAfter;
    }

    inline std::string toString() const {
        std::ostringstream str;
        str << inputShape << "x" << filterShape << " " << dataFormatToString(dataFormat) << " " << filterLayoutToString(filterLayout);
        if (algebra != Algebra::REAL)
            str << ", type " << algebra;
        if (stride != IntPair::ONES)
            str << ", stride " << stride.x << "," << stride.y;
        if (dilation != IntPair::ONES)
            str << ", dilation " << dilation.x << "," << dilation.y;
        if (groups != 1)
            str << ", " << groups << " groups";
        if (padBefore != IntPair::ZEROS || padAfter != IntPair::ZEROS)
            str << ", padding " << padBefore.x << "," << padBefore.y << ", " << padAfter.x << "," << padAfter.y;
        if (realValuedInput)
            str << ", real-valued";
        return str.str();
    }

    inline const Shape& getInputShape() const { return inputShape; }

    inline const Shape& getFilterShape() const { return filterShape; }

    inline const IntPair& getStride() const { return stride; }

    inline const IntPair& getDilation() const { return dilation; }

    inline int getGroups() const { return groups; }

    inline Algebra getAlgebra() const { return algebra; }

    inline DataFormat getDataFormat() const { return dataFormat; }

    inline FilterLayout getFilterLayout() const { return filterLayout; }

    inline bool isRealValuedInput() const { return realValuedInput; }

    inline IntPair getPaddingBefore() const { return padBefore; }

    inline IntPair getPaddingAfter() const { return padAfter; }
};


class Conv2DFwdDescriptor : public Conv2DDescriptor {
    const bool useBias;                 //!< if `true`, the bias addition is enabled
public:
    inline Conv2DFwdDescriptor(
        const Shape& inputShape,
        const Shape& filterShape,
        IntPair stride,
        IntPair dilation,
        Padding paddingPreset,
        const IntTuple& explicitPadding,
        int groups,
        Algebra algebra,
        DataFormat dataFormat,
        FilterLayout filterLayout,
        bool useBias,
        bool realValuedInput = false
    ):
        Conv2DDescriptor(inputShape, filterShape, stride, dilation, paddingPreset, explicitPadding, groups, algebra, dataFormat, filterLayout, realValuedInput),
        useBias(useBias)
    {}

    inline bool operator==(const Conv2DFwdDescriptor& another) const {
        return this->operator==(another) && useBias == another.useBias;
    }

    inline bool operator<(const Conv2DFwdDescriptor& another) const {
        int c = inputShape.compare(another.inputShape);
        if (c < 0) return false;
        if (c > 0) return true;

        c = filterShape.compare(another.filterShape);
        if (c < 0) return false;
        if (c > 0) return true;

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

        if (realValuedInput < another.realValuedInput) return true;
        if (another.realValuedInput < realValuedInput) return false;

        if (padBefore < another.padBefore) return true;
        if (another.padBefore < padBefore) return false;

        if (padAfter < another.padAfter) return true;
        if (another.padAfter < padAfter) return false;

        if (useBias < another.useBias) return true;
        if (another.useBias < useBias) return false;

        return false;
    }

    inline std::string toString() const {
        std::ostringstream str;
        str << Conv2DDescriptor::toString();
        if (useBias)
            str << ", with bias";
        return str.str();
    }

    inline bool isBiasUsed() const { return useBias; }

    inline const Shape getBiasShape() const {
        if (!useBias)
            return Shape::EMPTY;

        // for real-valued compute the bias is a vector of `channels` entries
        Conv2DFilterLayout filter(filterLayout, algebra);
        const int channels = filter.numOutputChannels(filterShape);
        if (algebra == Algebra::REAL)
            return Shape{ channels };

        // for other algebras it has 2 dimensions with the multivector dimension going outermost
        return Shape{ MULTIVECTOR_DIM[algebra], channels };
    }
};


class Conv2DBwdDescriptor : public Conv2DDescriptor {
    const bool requireInputGrad;        //!< if `true`, the input gradient is required

public:
    inline Conv2DBwdDescriptor(
        const Shape& inputShape,
        const Shape& filterShape,
        IntPair stride,
        IntPair dilation,
        Padding paddingPreset,
        const IntTuple& explicitPadding,
        int groups,
        Algebra algebra,
        DataFormat dataFormat,
        FilterLayout filterLayout,
        bool requireInputGrad = true,
        bool realValuedInput = false
    ):
        Conv2DDescriptor(inputShape, filterShape, stride, dilation, paddingPreset, explicitPadding, groups, algebra, dataFormat, filterLayout, realValuedInput),
        requireInputGrad(requireInputGrad)
    {}

    inline bool operator==(const Conv2DBwdDescriptor& another) const {
        return this->operator==(another) && requireInputGrad == another.requireInputGrad;
    }

    inline bool operator<(const Conv2DBwdDescriptor& another) const {
        int c = inputShape.compare(another.inputShape);
        if (c < 0) return false;
        if (c > 0) return true;

        c = filterShape.compare(another.filterShape);
        if (c < 0) return false;
        if (c > 0) return true;

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

        if (realValuedInput < another.realValuedInput) return true;
        if (another.realValuedInput < realValuedInput) return false;

        if (padBefore < another.padBefore) return true;
        if (another.padBefore < padBefore) return false;

        if (padAfter < another.padAfter) return true;
        if (another.padAfter < padAfter) return false;

        if (requireInputGrad < another.requireInputGrad) return true;
        if (another.requireInputGrad < requireInputGrad) return false;

        return false;
    }

    inline std::string toString() const {
        std::ostringstream str;
        str << Conv2DDescriptor::toString();
        if (requireInputGrad)
            str << ", with input gradient";
        return str.str();
    }

    inline bool isInputGradientRequired() const { return requireInputGrad; }
};


}