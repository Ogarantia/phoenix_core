/**
 * @file deferred_allocator.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Deferred device-dependent tensor allocation
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <map>
#include <mutex>

namespace upstride {

/**
 * @brief Deferred device-dependent tensor storage
 * Stores a bunch of tensors indexed by the device. Retrieves the tensor on a specific device reshaping it if a new shape is requested.
 * Transparently handles the destruction of all created tensors.
 * @tparam Device   the device type
 * @tparam T        tensor scalar type
 */
template <typename Device, typename T>
class DeferredAllocator {
   private:
    std::mutex mutex;
    std::map<const Device*, AllocatedTensor<Device, T>*> map;

   public:
    inline DeferredAllocator() {}

    inline ~DeferredAllocator() {
        for (auto& tensor : map)
            delete tensor.second;
    }

    /**
     * @brief Retrieves the tensor on a specific device reshaping it if the shape changed.
     * Creates a new one if no tensor found.
     * @param device    the device
     * @param shape     the requested shape
     * @param dirty     becomes `true` if the tensor is about to be allocated, `false` otherwise
     * @return the tensor
     */
    inline AllocatedTensor<Device, T>& get(const Device& device, const Shape& shape, bool& dirty) {
        std::lock_guard<std::mutex> lock(mutex);
        auto entry = map.find(&device);
        if (entry == map.end()) {
            // no yet tensor on the device, allocate new one
            AllocatedTensor<Device, T>* newbie = new AllocatedTensor<Device, T>(device, shape);
            map[&device] = newbie;
            dirty = true;
            return *newbie;
        } else {
            // the tensor exists for the given device; reshaping and returning it
            AllocatedTensor<Device, T>* tensor = entry->second;
            if (tensor->getShape() != shape) {
                tensor->reshape(shape);
                dirty = true;
            }
            else
                dirty = false;
            return *tensor;
        }
    }

    inline AllocatedTensor<Device, T>& get(const Device& device, const Shape& shape) {
        bool whatever;
        return get(device, shape, whatever);
    }
};

}  // namespace upstride