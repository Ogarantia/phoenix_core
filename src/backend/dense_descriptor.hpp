#pragma once

#include "../algebras.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "../utils.hpp"


namespace upstride {

class DenseDescriptor {
protected:
    const Shape inputShape;             //!< operation input tensor shape
    const Shape filterShape;            //!< operation filter tensor shape
    Algebra algebra;
    DataFormat dataFormat;
public:
    inline DenseDescriptor(const Shape& inputShape, const Shape& filterShape, Algebra algebra, DataFormat dataFormat):
        inputShape(inputShape), filterShape(filterShape), algebra(algebra), dataFormat(dataFormat)
    {
        //TODO: check if shapes match according to the data format
    }

    inline bool operator==(const DenseDescriptor& another) const {
        return algebra == another.algebra && dataFormat == another.dataFormat;
    }

    inline int getOutputChannels() const {
        // the outermost dimension in the filter tensor is multivector dimension
        const int idx = algebra == Algebra::REAL ? 0 : 1;
        switch (dataFormat) {
        case DataFormat::IO:
            return filterShape[idx + 1];
        case DataFormat::OI:
            return filterShape[idx];
        }
        throw std::runtime_error("Invalid data format");
    }
    
    inline Shape getInputShape() const { return inputShape; }

    inline Shape getFilterShape() const { return filterShape; }

    inline Shape getOutputShape() const {
        return Shape{ inputShape[0], getOutputChannels() };
    }

    inline Shape getBiasShape() const {
        if (algebra == Algebra::REAL)
            return Shape{ getOutputChannels() };
        return Shape{ MULTIVECTOR_DIM[algebra], getOutputChannels() };
    }

    inline Algebra getAlgebra() const { return algebra; }

    inline DataFormat getDataFormat() const { return dataFormat; }

    inline std::string toString() const { return ""; }
};


class DenseFwdDescriptor : public DenseDescriptor {
    bool useBias;
public:
    inline DenseFwdDescriptor(const Shape& inputShape, const Shape& filterShape, Algebra algebra, DataFormat dataFormat, bool useBias): 
        DenseDescriptor(inputShape, filterShape, algebra, dataFormat), useBias(useBias) 
    {}

    inline bool operator==(const DenseFwdDescriptor& another) const {
        return this->operator==(another) && useBias == another.useBias;
    }

    inline bool operator<(const DenseFwdDescriptor& another) const {
        int c = inputShape.compare(another.inputShape);
        if (c < 0) return false;
        if (c > 0) return true;

        c = filterShape.compare(another.filterShape);
        if (c < 0) return false;
        if (c > 0) return true;

        if (algebra < another.algebra) return true;
        if (another.algebra < algebra) return false;

        if (dataFormat < another.dataFormat) return true;
        if (another.dataFormat < dataFormat) return false;

        if (useBias < another.useBias) return true;
        if (another.useBias < useBias) return true;
        
        return false;
    }
    
    inline bool isBiasUsed() const { return useBias; }
};


class DenseBwdDescriptor : public DenseDescriptor {
    bool requireInputGrad;
public:
    inline DenseBwdDescriptor(const Shape& inputShape, const Shape& filterShape, Algebra algebra, DataFormat dataFormat, bool requireInputGrad): 
        DenseDescriptor(inputShape, filterShape, algebra, dataFormat), requireInputGrad(requireInputGrad) 
    {}
    
    inline bool operator==(const DenseBwdDescriptor& another) const {
        return this->operator==(another) && requireInputGrad == another.requireInputGrad;
    }

    inline bool operator<(const DenseBwdDescriptor& another) const {
        int c = inputShape.compare(another.inputShape);
        if (c < 0) return false;
        if (c > 0) return true;

        c = filterShape.compare(another.filterShape);
        if (c < 0) return false;
        if (c > 0) return true;

        if (algebra < another.algebra) return true;
        if (another.algebra < algebra) return false;

        if (dataFormat < another.dataFormat) return true;
        if (another.dataFormat < dataFormat) return false;

        if (requireInputGrad < another.requireInputGrad) return true;
        if (another.requireInputGrad < requireInputGrad) return false;

        return false;
    }

    inline bool isInputGradientRequired() const { return requireInputGrad; }
};

} // end namespace