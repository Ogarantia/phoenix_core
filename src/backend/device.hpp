#pragma once
#include <mutex>
#include "backend.hpp"
#include "conv2d_descriptor.hpp"
#include "dense_descriptor.hpp"
#include "op_collections.hpp"
#include "operation.hpp"
#include "memory_request.hpp"

namespace upstride {
class Device {
    Device(const Device&) = delete;  // disable copying
private:
    Context& context;
    GlobalOpCollection allOps;                          //!< all operations ready to go on the current device
    OpCollection<Conv2DFwdDescriptor> conv2dFwdOps;     //!< dense forward ops (subset of allOps)
    OpCollection<Conv2DBwdDescriptor> conv2dBwdOps;     //!< conv2d backward ops (subset of allOps)
    OpCollection<DenseFwdDescriptor> denseFwdOps;       //!< dense forward ops (subset of allOps)
    OpCollection<DenseBwdDescriptor> denseBwdOps;       //!< dense backward ops (subset of allOps)
    std::mutex accessControl;

    void* workspace;                                    //!< memory buffer shared across all operations to store temporary data
    size_t workspaceSize;                               //!< size of the shared memory buffer

    virtual void* malloc(size_t size) = 0;
    virtual void free(void* memory) = 0;

public:
    Device(Context& context):
        context(context),
        conv2dFwdOps(allOps),
        conv2dBwdOps(allOps),
        denseFwdOps(allOps),
        denseBwdOps(allOps),
        workspace(nullptr), workspaceSize(0)
    {}

    virtual ~Device() {
        // the workspace is freed in subclasses
    }

    inline Context& getContext() { return context; }

    inline std::mutex& getAccessControl() { return accessControl; }

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

    /**
     * @brief Ensures a minimum size of the workspace memory buffer on the device.
     * This buffer is used to store temporary data while an operation is executed. It is intended to be accessed using MemoryRequest
     * If the device already offers the requested or larger amount of memory, this function has no effect (apart returning the pointer).
     * Otherwise, it reallocates a larger workspace buffer.
     * @param size      The requested buffer size in bytes
     * @return pointer to the workspace buffer.
     */
    inline void* requestWorkspaceMemory(size_t size) {
        if (size > workspaceSize) {
            UPSTRIDE_SAYS("Memory request causes a reallocation: available %lu MB, requested %lu MB",
                workspaceSize / (1 << 20), size / (1 << 20));
            free(workspace);
            workspaceSize = size;
            workspace = malloc(size);
        }
        return workspace;
    }

    /**
     * @brief Frees the workspace memory buffer.
     */
    inline void freeWorkspaceMemory() {
        free(workspace);
        workspace = nullptr;
        workspaceSize = 0;
    }

    /**
     * @brief Returns the size in bytes of the workspace memory buffer allocated on the device.
     */
    inline size_t getWorkspaceSize() const {
        return workspaceSize;
    }

    /**
     * @brief Returns the device pointer alignment constraint in bytes.
     */
    virtual size_t getAlignmentConstraint() const {
        return 1;
    }

};
}