#pragma once
#include "isolated_thread.hpp"
#include "../backend.hpp"
#include "../device.hpp"

namespace upstride {
namespace device {
class CPU : public IsolatedThread, public Device {
   public:
    CPU(Context& context): Device(context) {}
    ~CPU() {
        freeWorkspaceMemory();
    }

    void* malloc(size_t size) override;
    void free(void* memory) override;
};
}  // namespace device
}  // namespace upstride