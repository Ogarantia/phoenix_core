#include "context.hpp"

using namespace upstride::cudnn;

const int Context::MAX_BLOCK_DEPTH = 64;      //!< maximum number of CUDA threads per block along Z dimension

upstride::device::CUDA& Context::registerDevice(const cudaStream_t& stream) {
    std::lock_guard<std::mutex> lock(mutex);
    auto entry = devices.find(stream);
    if (devices.find(stream) == devices.end()) {
        return devices.emplace(std::piecewise_construct, std::forward_as_tuple(stream), std::forward_as_tuple(*this, stream)).first->second;
    }
    else {
        return entry->second;
    }
}

void Context::cleanUp() {
    UPSTRIDE_SAYS("Cleaning up cuDNN context");
    std::lock_guard<std::mutex> lock(mutex);
    devices.clear();
}