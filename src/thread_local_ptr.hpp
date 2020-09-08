/**
 * @file thread_local_ptr.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Thread-local pointer class
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <exception>
#include <map>
#include <thread>
#include <mutex>

namespace upstride {

/**
 * @brief Handles a thread-local pointer to an object.
 * Object instances are created in every thread in a deferred fashion (when the operator (..) is called) and are
 * transparently accessed from different threads. The destruction is performed together with the pointer destruction.
 * @tparam Object   The object class
 */
template <class Object>
class ThreadLocalPtr {

    /**
     * @brief Retrieves the object in the storage by its local-thread pointer.
     * If no object is found, an exception is thrown.
     * @param pointer   The local-thread pointer
     * @return Object owned by the pointer.
     */
    inline Object* get() const {
        auto entry = storage.find(std::this_thread::get_id());
        if (entry == storage.end())
            throw std::runtime_error("The requested object is not initialized in the current thread");
        return entry->second;
    }

    // mapping of thread ids to objects instances
    std::map<std::thread::id, Object*> storage;
    std::mutex storageAccess;

   public:
    inline ThreadLocalPtr() {}

    inline ~ThreadLocalPtr() {
        for (auto entry : storage)
            delete entry.second;
    }

    /**
     * @brief Instantiates the object in the current thread if it does not already exist.
     * If the object was created before, the parameters not taken into account.
     * @param args Arguments to pass to the Object class constructor
     * @return Object& newly created or previously existing object
     */
    template <typename... Args>
    Object& operator()(Args&&... args) {
        std::lock_guard<std::mutex> lock(storageAccess);
        const auto id = std::this_thread::get_id();
        auto entry = storage.find(id);
        if (entry == storage.end())
            return *(storage[id] = new Object(args...));
        else
            return *entry->second;
    }

    inline Object* operator->() {
        return get();
    }

    inline Object& operator*() {
        return *get();
    }

    inline const Object* operator->() const {
        return get();
    }

    inline const Object& operator*() const {
        return *get();
    }
};

}  // namespace upstride