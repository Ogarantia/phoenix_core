#pragma once
#include <unordered_map>

#include "../backend.hpp"
#include "dnnl.hpp"

namespace upstride {

namespace onednn {

/**
 * @brief Converts a common data type to oneDNN data type handle
 * @tparam T    The input type
 * @return oneDNN data type
 */
template <typename T>
static inline dnnl::memory::data_type getDataType();

template <>
inline dnnl::memory::data_type getDataType<float>() { return dnnl::memory::data_type::f32; }

/**
 * @brief Retrieves oneDNN memory format tag corresponding to a given data format.
 * @param df the data format.
 * @return dnnl::memory::format_tag 
 */
static inline dnnl::memory::format_tag dataFormatToFormatTag(DataFormat df) {
    switch (df) {
        case DataFormat::NCHW:
            return dnnl::memory::format_tag::nchw;
        case DataFormat::NHWC:
            return dnnl::memory::format_tag::nhwc;
        case DataFormat::NC:
            return dnnl::memory::format_tag::nc;
        case DataFormat::CN:
            return dnnl::memory::format_tag::cn;
        default:
            throw std::invalid_argument("Unimplemented valid DataFormat.");
    }
}

/**
 * @brief Handy conversion of upstride::Shape to oneDNN memory dims
 * @param shape The shape to convert
 * @return dnnl::memory::dims of the given shape
 */
static inline dnnl::memory::dims shapeToDims(const Shape& shape) {
    return dnnl::memory::dims(shape.getShapePtr(), shape.getShapePtr() + shape.getSize());
}

/**
 * @brief OneDNN-specific shareable singleton context
 */
class Context : public upstride::Context {
    dnnl::engine oneEngine;
    dnnl::stream oneStream;
   private:
    inline void cleanUp() {}

   public:
    Context() : oneEngine(dnnl::engine::kind::cpu, 0), oneStream(oneEngine) {}

    /**
     * @brief Retrieves oneDNN engine instance associated with the current context.
     * @return a reference to a dnnl::engine object.
     */
    const dnnl::engine& getEngine() const { return oneEngine; }
    dnnl::engine& getEngine() { return oneEngine; }

    /**
     * @brief Executes a oneDNN operation primitive.
     * @param prim      The operation primitive to execute
     * @param args      A map of arguments (tensors) to pass to the operation
     */
    void execute(dnnl::primitive& prim, std::unordered_map<int, dnnl::memory>&& args);
};

}  // namespace onednn
}  // namespace upstride
