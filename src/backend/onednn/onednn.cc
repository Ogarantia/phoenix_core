
#include "context.hpp"
#include "dnnl.hpp"

using namespace upstride::onednn;

Context& Context::getInstance() {
    static upstride::onednn::Context context(1);
    return context;
}

void Context::execute(dnnl::primitive& prim, std::unordered_map<int, dnnl::memory>&& args) {
    prim.execute(oneStream, args);
    oneStream.wait();
}