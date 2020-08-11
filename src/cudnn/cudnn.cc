#include "context.hpp"

using namespace upstride::cudnn;

Context& Context::getInstance() {
    static upstride::cudnn::Context context(1);
    return context;
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