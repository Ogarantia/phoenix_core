#pragma once

#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "dense_descriptor.hpp"
#include "op_collections.hpp"
#include "../operation.hpp"

namespace upstride {
class Device {
    GlobalOpCollection allOps;                          //!< all operations ready to go on the current device
    OpCollection<Conv2DFwdDescriptor> conv2dFwdOps;     //!< dense forward ops (subset of allOps)
    OpCollection<Conv2DBwdDescriptor> conv2dBwdOps;     //!< conv2d backward ops (subset of allOps)
    OpCollection<DenseFwdDescriptor> denseFwdOps;       //!< dense forward ops (subset of allOps)
    OpCollection<DenseBwdDescriptor> denseBwdOps;       //!< dense backward ops (subset of allOps)

public:
    Device(Context& context):
        conv2dFwdOps(context, allOps),
        conv2dBwdOps(context, allOps),
        denseFwdOps(context, allOps),
        denseBwdOps(context, allOps)
    {}

    template<class Conv2DOperationClass>
    inline Conv2DOperationClass& getConv2DFwdOperation(const Conv2DFwdDescriptor& descriptor) {
        return conv2dFwdOps.get<Conv2DOperationClass>(descriptor);
    }

    template<class Conv2DOperationClass>
    inline Conv2DOperationClass& getConv2DBwdOperation(const Conv2DBwdDescriptor& descriptor) {
        return conv2dBwdOps.get<Conv2DOperationClass>(descriptor);
    }

    template<class DenseOperationClass>
    inline DenseOperationClass& getDenseFwdOperation(const DenseFwdDescriptor& descriptor) {
        return denseFwdOps.get<DenseOperationClass>(descriptor);
    }

    template<class DenseOperationClass>
    inline DenseOperationClass& getDenseBwdOperation(const DenseBwdDescriptor& descriptor) {
        return denseBwdOps.get<DenseOperationClass>(descriptor);
    }
};
}