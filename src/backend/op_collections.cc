#include "op_collections.hpp"
#include "device.hpp"

using namespace upstride;

void GlobalOpCollection::gc(Device& device, GarbageCollectingPolicy policy) {
    switch (policy) {
        // remove everything
        case GarbageCollectingPolicy::FLUSH:
            // drop operations
            for (auto op: list)
                delete op;
            list.clear();
            for (auto collection : collections)
                collection->update();
            // free the workspace
            device.freeWorkspaceMemory();
            break;

        // keeps top N=50 operations of the list (the most recently used ones)
        case GarbageCollectingPolicy::KEEP_TOP_100: {
            // skip first N-1 elements
            auto it = list.cbegin();
            for (size_t i = 1; i < 100 && it != list.cend(); ++i)
                it++;

            // check if Nth and (N+1)th ones exist
            if (it != list.cend() && std::next(it) != list.cend()) {
                // destroy operations starting from the (N+1)th
                for (auto j = std::next(it); j != list.cend(); ++j)
                    delete *j;

                // truncate the list after Nth element
                list.erase_after(it);

                // synchronize the contents list => map
                for (auto collection : collections)
                    collection->update();
            }
            break;
        }

        default:
            throw std::invalid_argument("Invalid garbage collection policy");
    }
}