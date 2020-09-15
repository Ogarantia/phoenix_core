
#include "context.hpp"
#include "dnnl.hpp"

using namespace upstride::onednn;

void Context::execute(dnnl::primitive& prim, std::unordered_map<int, dnnl::memory>&& args) {
    prim.execute(oneStream, args);
    oneStream.wait();
}