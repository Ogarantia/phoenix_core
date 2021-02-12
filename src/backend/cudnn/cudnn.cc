#include "context.hpp"

using namespace upstride::cudnn;

const unsigned int Context::MAX_BLOCK_DEPTH = 64;      //!< maximum number of CUDA threads per block along Z dimension

upstride::device::CUDA& Context::registerDevice(const cudaStream_t& stream) {
    std::lock_guard<std::mutex> lock(mutex);
    auto entry = devices.find(stream);
    if (entry == devices.end()) {
        return devices.emplace(std::piecewise_construct, std::forward_as_tuple(stream), std::forward_as_tuple(*this, stream)).first->second;
    }
    else {
        return entry->second;
    }
}


void Context::cleanUp() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& device : devices)
        device.second.enableCudnnHandleDestruction();
    devices.clear();
}