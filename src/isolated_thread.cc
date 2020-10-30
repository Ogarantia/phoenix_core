#include "isolated_thread.hpp"

using namespace upstride;

void IsolatedThread::threadFunc() {
    while (isRunning) {
        // wait for a message to process
        std::unique_lock<std::mutex> lock(access);
        while (messages.empty() && isRunning)
            threadNotifier.wait(lock);
            // . <--- queue access is locked at this point

        if (!messages.empty()) {
            // pick the topmost message from the queue
            Callable& message = *messages.front();

            // process it: a message is a Callable, so call it and catch exception if thrown
            try {
                message();
            }
            catch (...) {
                exceptions.emplace(processedMsgNumber + 1, std::current_exception());
            }

            // destroy the message and remove its pointer from the queue 
            delete messages.front();
            messages.pop_front();

            // increase number of processed messages and notify the callers
            processedMsgNumber++;

            // release the queue access
            lock.unlock();

            // notify callers about the processed message
            callerNotifier.notify_all();
        }

        else {
            // release the queue access
            lock.unlock();
        }
    }
}


IsolatedThread::IsolatedThread():
    isRunning(true),
    submittedMsgNumber(0), processedMsgNumber(0),
    thread(&IsolatedThread::threadFunc, this)
{}


IsolatedThread::~IsolatedThread() {
    isRunning = false;
    threadNotifier.notify_one();
    thread.join();
}


void IsolatedThread::call(Callable* entry) {
    // submit the message and keep its corresponding number
    std::unique_lock<std::mutex> lock(access);
    messages.push_back(entry);
    const int myMessageNumber = ++submittedMsgNumber;
    lock.unlock();

    // notify the managing thread about a new message
    threadNotifier.notify_one();

    // wait until the message is processed
    lock.lock();
    while (processedMsgNumber - myMessageNumber < 0)
        callerNotifier.wait(lock);

    // if an exception was thrown, rethrow
    auto it = exceptions.find(myMessageNumber);
    if (it != exceptions.end()) {
        lock.unlock();
        auto exception = it->second;
        exceptions.erase(it);
        std::rethrow_exception(exception);
        return;
    }

    lock.unlock();
}