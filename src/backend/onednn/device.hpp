#pragma once
#include "isolated_thread.hpp"
#include "../backend.hpp"
#include "../device.hpp"

namespace upstride {
namespace device {
class CPU : public IsolatedThread, public Device {
   public:
    CPU(Context& context): Device(context) {}
};
}  // namespace device
}  // namespace upstride