#include "kernels_utils.hpp"

namespace upstride {
namespace cuda {


std::string ConvDesc::toString() const {
    std::string descStr = "("
        + std::to_string(inputChannels) + ", "
        + std::to_string(outputChannels) + ", "
        + std::to_string(imageSize) + ", "
        + std::to_string(batchSize) + ")";

    return descStr;
}


// Convolution kernels Cache

bool ConvKernelsCache::checkCache(
    const ConvType convType, const ConvDesc& convDesc, PerfResult& optimalConf
) {
    std::lock_guard<std::mutex> lock(accessControl);

    if (convType == ConvType::invalid) {
        throw std::invalid_argument("Invalid convolution type");
    }

    // loop through the cache to try to find a suitable entry
    for (const auto& entry : cachedConfigurations[convType]) {
        // check if convolution descriptors match
        if (entry.first == convDesc) {
            optimalConf = entry.second;
            return true;
        }
    }

    return false;
}


void ConvKernelsCache::addToCache(
    const ConvType convType, const ConvDesc& convDesc, const PerfResult& optimalConf
) {
    std::lock_guard<std::mutex> lock(accessControl);

    if (convType == ConvType::invalid) {
        throw std::invalid_argument("Invalid convolution type");
    }

    // add cache entry
    cachedConfigurations[convType].push_back({convDesc, optimalConf});
}

}
}