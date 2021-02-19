#pragma once
#include <cstdint>
#include <stdexcept>
#include "operation.hpp"

namespace upstride {

// forward declarations
class MemoryRequest;


/**
 * @brief Temporary memory allocation interface
 */
class Allocator {
public:
    /**
     * @brief Allocates a temporary memory buffer.
     * The buffer is recycled after the operation is performed. No pointer shall be held outside of a run scope.
     * @param size      Buffer size in bytes
     * @return void*    buffer starting address
     */
    virtual void* mallocTemp(size_t size) = 0;

    /**
     * @brief Returns the device pointer alignment constraint in bytes.
     */
    virtual size_t getAlignmentConstraint() const = 0;
};


/**
 * @brief Points to an address in memory.
 * Cannot be used if the memory request that issued the pointer is valid.
 */
class Pointer {
    friend class MemoryRequest;

private:
    size_t shift;
    MemoryRequest* request;

    inline Pointer(MemoryRequest& request, size_t shift): request(&request), shift(shift) {}

public:
    Pointer(const Pointer&) = default;
    Pointer(Pointer&&) = default;
    Pointer& operator=(const upstride::Pointer&) = default;

    inline Pointer(): request(nullptr), shift(0) {}

    void* data();
    operator void*() { return data(); }

    template<typename T>
    inline T* cast() { return static_cast<T*>(data()); }

    inline Pointer operator+(size_t shift) const {
        return Pointer(*request, this->shift + shift);
    }

    inline Pointer operator-(size_t shift) const {
        if (shift > this->shift)
            throw std::runtime_error("Invalid pointer shift");
        return Pointer(*request, this->shift - shift);
    }
};


/**
 * @brief Requests memory to an allocator.
 */
class MemoryRequest {
    friend class Pointer;

private:
    Allocator& allocator;
    Operation& operation;   //!< operation the request is bound to
    uint8_t* address;
    size_t size;

public:
    inline MemoryRequest(Allocator& allocator, Operation& operation):
        allocator(allocator), operation(operation), address(nullptr), size(0)
    {}

    inline ~MemoryRequest() {
        operation.updateMemoryNeeds(size);
    }

    /**
     * @brief Issues a new pointer.
     * Does not perform any allocation but stores the necessary information submitted to the allocator later.
     * @param size          Size in bytes of a memory to allocate
     * @return a Pointer instance allowing to access the memory once the request is submitted.
     */
    Pointer alloc(size_t size);

    /**
     * @brief Checks if the request is valid, i.e., if all pointers it has issued point to valid usable memory addresses.
     * A request is valid if it has been submitted to the allocator, or if it is of zero size.
     * @return true 
     * @return false 
     */
    inline bool isValid() const {
        return address != nullptr || size == 0;
    }

    /**
     * @brief Submits the memory request to the allocator so that it can provide the necessary memory.
     * All the pointers issued from the current request are "materialized", i.e., they receive valid addresses and can be used as regular pointers.
     */
    void submit();
};

}