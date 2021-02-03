#pragma once
#include <map>
#include <forward_list>
#include <iterator>
#include <vector>
#include <sstream>
#include "operation.hpp"
#include "backend.hpp"

namespace upstride {

typedef std::forward_list<Operation*> OperationsList;


/**
 * @brief Garbage collection policy specification.
 */
enum class GarbageCollectingPolicy {
    FLUSH,          //!< recycles all the existing operation instances
    KEEP_TOP_100
};


/**
 * @brief Operations collection interface
 */
class OpCollectionInterface {
public:
    virtual void update() = 0;
};


template <class Descriptor>
class OpCollection;

class Device;


/**
 * @brief Collection containing a list of operations of all kinds.
 * Used to perform operation kind-agnostic stuff like shared memory management and garbage collection.
 */
class GlobalOpCollection {
    template <class Descriptor>
    friend class OpCollection;

    private:
        OperationsList list;                                //!< list of operations having the most recently used ones in front
        std::vector<OpCollectionInterface*> collections;    //!< op-specific collections tied to this global collection

        /**
         * @brief Binds an operation-specific collection to the current global op collection.
         * @param collection        The operation-specific collection
         */
        inline void bindCollection(OpCollectionInterface& collection) {
            collections.push_back(&collection);
        }

    public:
        inline GlobalOpCollection() {}

        /**
         * @brief Runs garbage collection.
         * @param device        A device instance
         * @param policy        Garbage collection policy defining which instances are recycled.
         */
        void gc(Device& device, GarbageCollectingPolicy policy);

        /**
         * @brief Returns the list of all operations.
         * @return OperationsList&
         */
        inline OperationsList& getList() {
            return list;
        }
};


/**
 * @brief Collection of operations of a specific kind.
 * Retrieves an operation instance according to its description (creates a new one if not found).
 * @tparam Descriptor    Operation descriptor class presenting a complete description of the operation
 */
template <class Descriptor>
class OpCollection : public OpCollectionInterface {
    private:
        GlobalOpCollection& allOps;                 //!< global operation collection
        std::map<Descriptor, Operation*> map;       //!< descriptor => operation mapping used for fast search by descriptor

        /**
         * @brief Updates the map according to the global cache contents.
         */
        inline void update() override {
            // reconstruct the inverse map first (op ptr => descriptor)
            std::map<Operation*, Descriptor> inverseMap;
            for (auto it: map)
                inverseMap.emplace(it.second, it.first);

            // rebuild the main map taking keys from the inverse map and values from the list
            map.clear();
            for (auto op: allOps.getList()) {
                auto it = inverseMap.find(op);
                if (it != inverseMap.end())
                    map.emplace(it->second, op);
            }
        }

    public:
        OpCollection(GlobalOpCollection& allOps): allOps(allOps) {
            // attach this collection to the global ops collection to have the GC working
            allOps.bindCollection(*this);
        }

        /**
         * @brief Returns an operation instance according to a given descriptor.
         * @tparam Device           Device class
         * @tparam OpClass          The required operation type
         * @param descriptor        The operation descriptor
         * @return Operation&       The operation instance
         */
        template<class Device, class OpClass>
        inline OpClass& get(Device& device, const Descriptor& descriptor) {
            // check if a corresponding operation instance is already available
            auto it = map.find(descriptor);
            if (it != map.end()) {
                // an instance is found; put it on top of the list as the most recently used one
                Operation* op = it->second;
                auto& list = allOps.getList();
                if (list.front() != op) {
                    list.remove(op);
                    list.push_front(op);
                }
                return static_cast<OpClass&>(*op);
            }

            // run garbage collection with a default policy (for debugging purposes)
            allOps.gc(device, GarbageCollectingPolicy::KEEP_TOP_100);

            // no operation instance available; create a new one
            OpClass* newOp = new OpClass(device, descriptor);
            map.emplace(descriptor, newOp);
            allOps.getList().push_front(newOp);
            return *newOp;
        }


        inline std::string toString() const {
            std::ostringstream str;
            std::map<Operation*, Descriptor> inverseMap;
            for (auto it: map)
                inverseMap.emplace(it.second, it.first);

            int ctr = 0;
            for (auto op: allOps.getList()) {
                const auto it = inverseMap.find(op);
                if (it != inverseMap.cend())
                    str << ++ctr << ": " << it->second.toString() << std::endl;
            }
            return str.str();
        }
};


}