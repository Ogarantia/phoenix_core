#pragma once
#include <map>
#include <forward_list>
#include <iterator>
#include "backend.hpp"

namespace upstride {

template <class Descriptor, class Operation>
class OperationsCache {
    private:
        Context& context;
        std::map<Descriptor, Operation*> cacheMap;      //!< descriptor => operation mapping used for fast search by descriptor
        std::forward_list<Operation*> cacheList;        //!< list of operations having the most recently used ones in front; for fast garbage collection

        /**
         * @brief Updates the cache map according to the contents of the cache list.
         */
        inline void updateMap() {
            // reconstruct the inverse map first (op ptr => descriptor)
            std::map<Operation*, Descriptor> inverseMap;
            for (auto it: cacheMap)
                inverseMap.emplace(it.second, it.first);

            // rebuild the main map taking keys from the inverse map and values from the list
            cacheMap.clear();
            for (auto op: cacheList) {
                auto it = inverseMap.find(op);
                cacheMap.emplace(it->second, op);
            }
        }

    public:
        /**
         * @brief Garbage collection policy specification.
         */
        enum class GarbageCollectingPolicy {
            FLUSH,          //!< recycles all the existing operation instances
            KEEP_TOP_50
        };

        OperationsCache(Context& context): context(context) {}

        /**
         * @brief Returns an operation instance according to a given descriptor.
         * @tparam OpClass          The required operation type
         * @param descriptor        The operation descriptor
         * @return Operation&       The operation instance
         */
        template<class OpClass>
        inline OpClass& get(const Descriptor& descriptor) {
            // check if a corresponding operation instance is already available
            auto it = cacheMap.find(descriptor);
            if (it != cacheMap.end()) {
                // an instance is found; put it on top of the list as the most recently used one
                Operation* op = it->second;
                if (cacheList.front() != op) {
                    cacheList.remove(op);
                    cacheList.push_front(op);
                }
                return static_cast<OpClass&>(*op);
            }

            // run garbage collection with a default policy (for debugging purposes)
            gc(GarbageCollectingPolicy::KEEP_TOP_50);

            // no operation instance available; create a new one
            OpClass* newOp = new OpClass(context, descriptor);
            cacheMap.emplace(descriptor, newOp);
            cacheList.push_front(newOp);
            return *newOp;
        }


        /**
         * @brief Runs garbage collection.
         *
         * @param policy        Garbage collection policy defining which instances are recycled.
         */
        inline void gc(GarbageCollectingPolicy policy) {
            switch (policy) {
                // remove everything
                case GarbageCollectingPolicy::FLUSH:
                    cacheMap.clear();
                    for (auto op: cacheList)
                        delete op;
                    cacheList.clear();
                    break;

                // keeps top N=50 operations of the list (the most recently used ones)
                case GarbageCollectingPolicy::KEEP_TOP_50: {
                    // skip first N-1 elements
                    auto it = cacheList.cbegin();
                    for (size_t i = 0; i < 49 && it != cacheList.cend(); ++i)
                        it++;

                    // check if Nth and (N+1)th ones exist
                    if (it != cacheList.cend() && std::next(it) != cacheList.cend()) {
                        // destroy operations starting from the (N+1)th
                        for (auto j = std::next(it); j != cacheList.cend(); ++j)
                            delete *j;

                        // truncate the list after Nth element
                        cacheList.erase_after(it);

                        // synchronize the contents list => map
                        updateMap();
                    }
                    break;
                }

                default:
                    throw std::invalid_argument("Invalid garbage collection policy");
            }
        }
    };

}