#pragma once
#include "memory_request.hpp"

namespace upstride {

template<typename Device, typename T>
class TemporaryTensor : public Tensor<Device, T> {
private:
    Pointer ptr;

public:
    TemporaryTensor(Device& device, MemoryRequest& request, const Shape& shape):
        Tensor<Device, T>(device, shape, nullptr),
        ptr(request.alloc(shape.numel() * sizeof(T)))
    {}


    inline void prepare() {
        this->tensor = ptr.cast<T>();
    }

    static void prepare(std::initializer_list<TemporaryTensor<Device, T>&> tensors) {
        for (auto& t : tensors)
            t.prepare();
    }
};

}