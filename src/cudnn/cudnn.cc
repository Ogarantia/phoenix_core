#include "context.hpp"

using namespace upstride::cudnn;

Context& Context::getInstance() {
    static upstride::cudnn::Context context(1);
    return context;
}

Memory::Memory(size_t sizeBytes): size(sizeBytes) {
    Context::getInstance().raiseIfError(
        cudaMalloc(&ptr, sizeBytes)
    );
}

Memory::Memory(Memory&& another): size(another.size), ptr(another.ptr) {
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


void Memory::free() {
    Context::getInstance().raiseIfError(cudaFree(ptr));
    ptr = nullptr;
    size = 0;
}