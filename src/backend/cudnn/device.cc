#include "device.hpp"
#include "context.hpp"


namespace upstride {
namespace device {

// GPU querying

/**
 * @brief Get the number of registers available per thread block on the specified device
 *
 * @param dev                               the device to be queried for the available registers
 */
int getDeviceRegistersPerThreadBlock(int dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudnn::Context::raiseIfError("getDeviceRegistersPerThreadBlock cudaGetDeviceProperties failed");
    return deviceProp.regsPerBlock;
}


int CUDA::getRegistersPerThreadBlock(bool acrossAllDevices) {
    int registersPerThreadBlock {0};

    if (acrossAllDevices) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        cudnn::Context::raiseIfError("getRegistersPerThreadBlock cudaGetDeviceCount failed");
        if (deviceCount == 0) {
            throw std::logic_error("No available devices that support CUDA detected");
        }
        // compute minimum of the available registers across the devices
        registersPerThreadBlock = getDeviceRegistersPerThreadBlock(0);
        for (int dev = 1; dev < deviceCount; ++dev) {
            registersPerThreadBlock = std::min(registersPerThreadBlock, getDeviceRegistersPerThreadBlock(dev));
        }

    } else {
        int currentDevice {0};
        cudaGetDevice(&currentDevice);
        cudnn::Context::raiseIfError("getRegistersPerThreadBlock cudaGetDevice failed");
        registersPerThreadBlock = getDeviceRegistersPerThreadBlock(currentDevice);
    }
    return registersPerThreadBlock;
}

}
}