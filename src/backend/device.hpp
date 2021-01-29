#pragma once

#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "operations_cache.hpp"
#include "../operation.hpp"

namespace upstride {
class Device {
    OperationsCache<Conv2DDescriptor, Operation> conv2dCache;
    public:
        Device(Context& context): conv2dCache(context) {}

        template<class Conv2DOperationClass>
        inline Conv2DOperationClass& getConv2DOperation(const Conv2DDescriptor& descriptor) {
            return conv2dCache.get<Conv2DOperationClass>(descriptor);
        }
};
}