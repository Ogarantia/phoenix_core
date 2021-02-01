#pragma once

#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "op_collections.hpp"
#include "../operation.hpp"

namespace upstride {
class Device {
    GlobalOpCollection allOps;
    OpCollection<Conv2DDescriptor> conv2dFwdOps;
    public:
        Device(Context& context): conv2dFwdOps(context, allOps) {}

        template<class Conv2DOperationClass>
        inline Conv2DOperationClass& getConv2DOperation(const Conv2DDescriptor& descriptor) {
            return conv2dFwdOps.get<Conv2DOperationClass>(descriptor);
        }
};
}