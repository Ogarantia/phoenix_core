#include "context.hpp"

using namespace upstride::cudnn;

const int Context::MAX_BLOCK_DEPTH = 64;      //!< maximum number of CUDA threads per block along Z dimension

upstride::device::CUDA& Context::registerDevice(const cudaStream_t& stream) {
    std::lock_guard<std::mutex> lock(mutex);
    auto entry = devices.find(stream);
    if (devices.find(stream) == devices.end()) {
        return devices.emplace(stream, stream).first->second;
    }
    else {
        return entry->second;
    }
}

void Context::cleanUp() {
    UPSTRIDE_SAYS(*this, "Clean up cuDNN context");
    std::lock_guard<std::mutex> lock(mutex);
    devices.clear();
}

Memory::Memory(size_t sizeBytes) : size(sizeBytes) {
    auto status = cudaMalloc(&ptr, sizeBytes);
    if (status != cudaError::cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

Memory::Memory(Memory&& another) : size(another.size), ptr(another.ptr) {
    another.ptr = nullptr;
    another.size = 0;
}

Memory& Memory::operator=(Memory&& another) {
    free();
    ptr = another.ptr;
    size = another.size;
    another.ptr = nullptr;
    another.size = 0;
    return *this;
}

Memory::~Memory() {
    cudaFree(ptr);
}

void Memory::zero() {
    auto status = cudaMemset(ptr, 0, size);
    if (status != cudaError::cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void Memory::free() {
    auto status = cudaFree(ptr);
    if (status != cudaError::cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    ptr = nullptr;
    size = 0;
}