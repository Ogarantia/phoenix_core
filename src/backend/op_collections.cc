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

        // keeps top N=100 operations of the list (the most recently used ones) if the list size is bigger than 200
        case GarbageCollectingPolicy::KEEP_TOP_100: {
            // skip first N-1 elements
            auto it = list.cbegin();
            size_t i = 1;
            size_t maxMemoryNeeds = 0;
            for (; i < 100 && it != list.cend(); ++i) {
                maxMemoryNeeds = std::max((*it)->getMemoryNeeds(), maxMemoryNeeds);
                it++;
            }

            // check if Nth and (N+1)th ones exist
            if (it != list.cend() && std::next(it) != list.cend()) {
                // check if there are at least 200 elements in the list
                for (auto itCount = it; i <= 200; i++, itCount++)
                    if (itCount == list.cend())
                        return;

                // update memory needs with the last element (Nth)
                maxMemoryNeeds = std::max((*it)->getMemoryNeeds(), maxMemoryNeeds);

                // destroy operations starting from the (N+1)th
                for (auto j = std::next(it); j != list.cend(); ++j)
                    delete *j;

                // truncate the list after Nth element
                list.erase_after(it, list.cend());

                // synchronize the contents list => map
                for (auto collection : collections)
                    collection->update();

                // update the workspace memory (if ever it can be reduced in size)
                if (device.getWorkspaceSize() > maxMemoryNeeds) {
                    device.freeWorkspaceMemory();
                    device.requestWorkspaceMemory(maxMemoryNeeds);
                }
            }
            break;
        }

        default:
            throw std::invalid_argument("Invalid garbage collection policy");
    }
}