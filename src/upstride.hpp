#ifndef KERNEL_UPSTRIDE_OPS_H_
#define KERNEL_UPSTRIDE_OPS_H_

#include "utils.hpp"

namespace upstride {

namespace device {

typedef struct {
} CPU;
typedef struct {
} GPU;

}  // namespace device

class Context {
    const int typeDimensions;

   protected:
    Context(const int td) : typeDimensions(td){};
};

template <typename Device, typename T>
struct UpstrideConv2DFunctor {
    /**
     * @brief Computes regular D convolution
     * @param input     The input tensor
     * @param kernel    The tensor containing convolution filters
     * @param output    The output tensor
     * @param format    Dimension orders specification in input and output tensors
     */
    void operator()(const Tensor<const T>& input,
                    const Tensor<const T>& kernel,
                    Tensor<T>& output,
                    DataFormat format);
};

template <>
struct UpstrideConv2DFunctor<device::CPU, float> {
    class Backend;
    Backend* backend;

    UpstrideConv2DFunctor() : backend(nullptr) {}
    ~UpstrideConv2DFunctor();

    void operator()(const Tensor<const float>& input,
                    const Tensor<const float>& kernel,
                    Tensor<float>& output,
                    DataFormat format);
};

}  // namespace upstride

#endif  //KERNEL_TIME_TWO_H_
