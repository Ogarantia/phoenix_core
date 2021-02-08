#include "memory_request.hpp"
#include "device.hpp"

using namespace upstride;

void* Pointer::data() {
#ifdef UPSTRIDE_DEBUG
    if (!request)
        throw std::runtime_error("Trying to use a pointer without associated memory request");
    if (!request->isValid())
        throw std::runtime_error("Submit the memory request to a device before using pointers");
#endif
    return request->address + shift;
}


MemoryRequest::MemoryRequest(Device& device, Operation& operation):
    device(device), operation(operation), address(nullptr), size(0)
{}


Pointer MemoryRequest::alloc(size_t size) {
    Pointer result(*this, this->size);
    const size_t alignment = device.getAlignmentConstraint();
    this->size += ((size + alignment - 1) / alignment) * alignment;
    return result;
}


void MemoryRequest::submit() {
#ifdef UPSTRIDE_DEBUG
    // check if already submitted
    if (address)
        throw std::runtime_error("Trying to submit a memory request twice");
#endif
    // request the memory to the device and set the address to the beginning of the workspace
    address = static_cast<uint8_t*>(device.requestWorkspaceMemory(size));
}