#include "memory_request.hpp"
#include "device.hpp"

using namespace upstride;

void* Pointer::data() {
    if (!request.isValid())
        throw std::runtime_error("Submit the memory request to a device before using pointers");
    return request.address + shift;
}


void MemoryRequest::submit(Device& device) {
    // request the memory to the device and set the address to the beginning of the workspace
    address = static_cast<uint8_t*>(device.requestWorkspaceMemory(size));
}