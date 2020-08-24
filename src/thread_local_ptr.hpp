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

namespace upstride {

/**
 * @brief Handles a thread-local pointer to an object
 *
 * @tparam Object   The object class
 */
template <class Object>
class ThreadLocalPtr {
    /**
     * @brief Thread-local storage of objects
     * Wraps a mapping of thread-local pointers to the objects. Thread-local qualified to be destroyed when the thread exits.
     */
    class Storage {
       private:
        std::map<ThreadLocalPtr<Object>*, Object*> map;

       public:
        inline Storage() {}
        inline ~Storage() {
            clear();
        }

        /**
         * @brief Retrieves the object in the storage by its local-thread pointer.
         * If no object is found, an exception is thrown.
         * @param pointer   The local-thread pointer
         * @return Object owned by the pointer.
         */
        inline Object* get(ThreadLocalPtr<Object>* pointer) const {
            auto entry = map.find(pointer);
            if (entry == map.end())
                throw std::runtime_error("The requested object is not initialized in the current thread");
            return entry->second;
        }

        /**
         * @brief Creates a new object.
         * @param pointer       The local-thread pointer owning the new instance
         * @param args          Arguments to pass to the object constructor
         * @return newly created instance.
         */
        template <typename... Args>
        inline Object& put(ThreadLocalPtr<Object>* pointer, Args&&... args) {
            auto entry = map.find(pointer);
            if (entry == map.end())
                return *(map[pointer] = new Object(args...));
            return *entry->second;
        }

        /**
         * @brief Destroys all the objects in the storage.
         */
        inline void clear() {
            for (auto entry : map)
                delete entry.second;
            map.clear();
        }
    };

    static thread_local Storage storage;  //!< objects storage

   public:
    inline ThreadLocalPtr() {}

    inline ~ThreadLocalPtr() {
        storage.clear();
    }

    /**
     * @brief Instantiates the object in the current thread if it does not already exist.
     * If the object was created before, the parameters not taken into account.
     * @param args Arguments to pass to the Object class constructor
     * @return Object& newly created or previously existing object
     */
    template <typename... Args>
    Object& operator()(Args&&... args) {
        return storage.put(this, args...);
    }

    inline Object* operator->() {
        return storage.get(this);
    }

    inline Object& operator*() {
        return *storage.get(this);
    }

    inline const Object* operator->() const {
        return storage.get(this);
    }

    inline const Object& operator*() const {
        return *storage.get(this);
    }
};

template <class Object>
thread_local typename ThreadLocalPtr<Object>::Storage ThreadLocalPtr<Object>::storage;

}  // namespace upstride