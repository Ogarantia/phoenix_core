#pragma once
#include "memory_request.hpp"

namespace upstride {

template<typename Device, typename T>
class TemporaryTensor : public Tensor<Device, T> {
private:
    Pointer ptr;

public:
    TemporaryTensor<Device, T>& operator=(const TemporaryTensor<Device, T>& another) {
        if (&this->device != &another.device)
            throw std::runtime_error("Temporary tensor device mismatch");
        this->ptr = another.ptr;
        this->shape = another.shape;
        this->tensor = another.tensor;
        return *this;
    }

    TemporaryTensor(Device& device): Tensor<Device, T>(device, Shape(), nullptr) {}

    TemporaryTensor(Device& device, MemoryRequest& request, const Shape& shape):
        Tensor<Device, T>(device, shape, nullptr),
        ptr(request.alloc(shape.numel() * sizeof(T)))
    {}


    inline void prepare() {
        this->tensor = ptr.cast<T>();
    }
};

}