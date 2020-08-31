#include "context.hpp"

using namespace upstride::cudnn;

Context& Context::getInstance() {
    static upstride::cudnn::Context context;
    return context;
}


const upstride::device::CUDA& Context::registerDevice(const cudaStream_t& stream) {
    auto entry = devices.find(stream);
    if (devices.find(stream) == devices.end()) {
        return devices.emplace(stream, stream).first->second;
    }
    else {
        return entry->second;
    }
}