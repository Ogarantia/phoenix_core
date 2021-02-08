#pragma once

namespace upstride {

class MemoryRequest;

/**
 * @brief Base class for operations
 */
class Operation {
    friend class MemoryRequest;
private:
    size_t memoryNeeds;     //!< memory needs of the operation in bytes; largest ones ever known for the current instance

    inline void updateMemoryNeeds(size_t newValue) {
        if (newValue > memoryNeeds)
            memoryNeeds = newValue;
    }

public:
    Operation(): memoryNeeds(0) {}
    virtual ~Operation() {}
    inline size_t getMemoryNeeds() const { return memoryNeeds; }
};

}