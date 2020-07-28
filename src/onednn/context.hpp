#pragma once
#include <unordered_map>

#include "dnnl.hpp"
#include "upstride.hpp"

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
static inline dnnl::memory::format_tag convertDataFormatToFormatTag(DataFormat df) {
    switch (df) {
        case DataFormat::NCHW:
            return dnnl::memory::format_tag::nchw;
        case DataFormat::NHWC:
            return dnnl::memory::format_tag::nhwc;
        default:
            throw std::invalid_argument("Unimplemented valid DataFormat.");
    }
}

/**
 * @brief OneDNN-specific shareable singleton context
 */
class Context : public upstride::Context {
    dnnl::engine oneEngine;
    dnnl::stream oneStream;

    Context(const int typeDim) : upstride::Context(typeDim), oneEngine(dnnl::engine::kind::cpu, 0), oneStream(oneEngine) {}

   public:
    /**
     * @brief Provides the instance of oneDNN context.
     * @return the context.
     */
    static Context& getInstance();

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
