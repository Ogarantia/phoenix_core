#pragma once

#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "dense_descriptor.hpp"
#include "op_collections.hpp"
#include "../operation.hpp"

namespace upstride {
class Device {
private:
    Context& context;
    GlobalOpCollection allOps;                          //!< all operations ready to go on the current device
    OpCollection<Conv2DFwdDescriptor> conv2dFwdOps;     //!< dense forward ops (subset of allOps)
    OpCollection<Conv2DBwdDescriptor> conv2dBwdOps;     //!< conv2d backward ops (subset of allOps)
    OpCollection<DenseFwdDescriptor> denseFwdOps;       //!< dense forward ops (subset of allOps)
    OpCollection<DenseBwdDescriptor> denseBwdOps;       //!< dense backward ops (subset of allOps)

public:
    Device(Context& context):
        context(context),
        conv2dFwdOps(allOps),
        conv2dBwdOps(allOps),
        denseFwdOps(allOps),
        denseBwdOps(allOps)
    {}

    inline Context& getContext() { return context; }

    template<class DeviceClass, class OperationClass>
    inline OperationClass& getConv2DFwdOperation(const Conv2DFwdDescriptor& descriptor) {
        return conv2dFwdOps.get<DeviceClass, OperationClass>(static_cast<DeviceClass&>(*this), descriptor);
    }

    template<class DeviceClass, class OperationClass>
    inline OperationClass& getConv2DBwdOperation(const Conv2DBwdDescriptor& descriptor) {
        return conv2dBwdOps.get<DeviceClass, OperationClass>(static_cast<DeviceClass&>(*this), descriptor);
    }

    template<class DeviceClass, class OperationClass>
    inline OperationClass& getDenseFwdOperation(const DenseFwdDescriptor& descriptor) {
        return denseFwdOps.get<DeviceClass, OperationClass>(static_cast<DeviceClass&>(*this), descriptor);
    }

    template<class DeviceClass, class OperationClass>
    inline OperationClass& getDenseBwdOperation(const DenseBwdDescriptor& descriptor) {
        return denseBwdOps.get<DeviceClass, OperationClass>(static_cast<DeviceClass&>(*this), descriptor);
    }
};
}