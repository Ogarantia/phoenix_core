#pragma once
#include <map>
#include <forward_list>
#include "backend.hpp"

namespace upstride {

template <class Descriptor, class Operation>
class OperationsCache {
    private:
        Context& context;
        std::map<Descriptor, Operation*> cacheMap;      //!< descriptor => operation mapping used for fast search
        std::forward_list<Operation*> cacheList;       //!< list of available operations for fast garbage collection

        /**
         * @brief Updates the cache map according to the contents of the cache list.
         */
        inline void updateMap() {
            // reconstruct the inverse map first
            std::map<Operation*, Descriptor> inverseMap;
            for (auto it: cacheMap)
                inverseMap.emplace(it.second, it.first);

            // reconstruct the map
            cacheMap.clear();
            for (auto op: cacheList) {
                auto it = inverseMap.find(op);
                cacheMap[it->second] = op;
            }
        }

    public:
        enum class GarbageCollectingPolicy {
            FLUSH,          //!< recycles all the existing operation instances
            KEEP_TOP_50
        };

        OperationsCache(Context& context): context(context) {}

        /**
         * @brief Returns an operation instance according to a given descriptor.
         * @param descriptor        The descriptor
         * @return Operation&       The operation instance
         */
        template<class OpClass>
        inline OpClass& get(const Descriptor& descriptor) {
            // check if a corresponding operation instance is already available
            auto it = cacheMap.find(descriptor);
            if (it != cacheMap.end()) {
                // an instance is found; put it on top of the list as the most recently used one
                Operation* op = it->second;
                cacheList.remove(op);
                cacheList.push_front(op);
                return static_cast<OpClass&>(*op);
            }

            // run garbage collection with a default policy (for debugging purposes)
            gc(GarbageCollectingPolicy::KEEP_TOP_50);

            // no operation instance available; create a new one
            OpClass* newOp = new OpClass(context, descriptor);
            cacheMap[descriptor] = newOp;
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

                // keeps top 50 operations of the list (the most recently used ones)
                case GarbageCollectingPolicy::KEEP_TOP_50: {
                    auto it = cacheList.cbegin();
                    for (size_t i = 0; i < 49 && it != cacheList.cend(); ++i)
                        it++;
                    if (it != cacheList.cend()) {
                        auto next(it);
                        if (++next != cacheList.cend()) {
                            cacheList.erase_after(it);
                            updateMap();
                        }
                    }
                    break;
                }

                default:
                    throw std::invalid_argument("Invalid garbage collection policy");
            }
        }
    };

}