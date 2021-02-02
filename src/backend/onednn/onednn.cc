
#include "context.hpp"
#include "device.hpp"
#include "dnnl.hpp"

using namespace upstride;

void onednn::Context::execute(dnnl::primitive& prim, std::unordered_map<int, dnnl::memory>&& args) {
    prim.execute(oneStream, args);
    oneStream.wait();
}

void* device::CPU::malloc(size_t size) {
    return ::malloc(size);
}

void device::CPU::free(void* memory) {
    ::free(memory);
}