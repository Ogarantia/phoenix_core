#pragma once

#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "op_collections.hpp"
#include "../operation.hpp"

namespace upstride {
class Device {
    GlobalOpCollection allOps;                          //!< all operations ready to go on the current device
    OpCollection<Conv2DFwdDescriptor> conv2dFwdOps;     //!< conv2d forward ops (subset of allOps)
    OpCollection<Conv2DBwdDescriptor> conv2dBwdOps;     //!< conv2d backward ops (subset of allOps)

public:
    Device(Context& context):
        conv2dFwdOps(context, allOps),
        conv2dBwdOps(context, allOps)
    {}

    template<class Conv2DOperationClass>
    inline Conv2DOperationClass& getConv2DFwdOperation(const Conv2DFwdDescriptor& descriptor) {
        return conv2dFwdOps.get<Conv2DOperationClass>(descriptor);
    }

    template<class Conv2DOperationClass>
    inline Conv2DOperationClass& getConv2DBwdOperation(const Conv2DBwdDescriptor& descriptor) {
        return conv2dBwdOps.get<Conv2DOperationClass>(descriptor);
    }
};
}