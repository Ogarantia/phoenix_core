#include "context.hpp"

using namespace upstride::cudnn;

Context& Context::getInstance() {
    static upstride::cudnn::Context context;
    return context;
}