#pragma once

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

/**
 * @brief Operation functors declarations
 * The operations are only declared here. They are specialized further on for every backend.
 * @tparam Device       A device the specific implementation is designed for
 * @tparam T            A scalar datatype
 */
template <typename Device, typename T>
class UpstrideConv2DFunctor;

}  // namespace upstride
