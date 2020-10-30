/**
 * @file isolated_thread.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Executing callable messages in an isolated thread
 * @copyright Copyright (c) 2020 UpStride
 */


#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <stdexcept>

namespace upstride {

/**
 * @internal
 * @brief Internal mechanics of the tuple unwrapping.
 */
namespace internal {
    /**
     * @brief This recursive template unpacks the tuple of arguments into variadic template arguments until the counter
     * reaches 0, when the method is called.
     * @tparam N the argument number counter.
     */
    template <unsigned int N>
    struct TupleUnwrapper {
        template <class Object, typename... ArgsF, typename... ArgsT, typename... Args>
        static void applyTuple(Object* object, void (Object::*method)(ArgsF...), const std::tuple<ArgsT...>& argsTuple, Args&... args) {
            TupleUnwrapper<N - 1>::applyTuple(object, method, argsTuple, std::get<N - 1>(argsTuple), args...);
        }
    };

    /**
     * @brief The recursive unwrapping endpoint calling the given method of an object.
     */
    template <>
    struct TupleUnwrapper<0> {
        template <class Object, typename... ArgsF, typename... ArgsT, typename... Args>
        static void applyTuple(Object* object, void (Object::*method)(ArgsF...), const std::tuple<ArgsT...>&, Args&... args) {
            (object->*method)(args...);
        }
    };
}


/**
 * @brief Calls a method of a specific object with arguments stored in an std::tuple
 * @param object        The object class
 * @param method        The method to call
 * @param argsTuple     The tuple of arguments
 */
template <class Object, typename... ArgsF, typename... ArgsT> 
void applyTuple(Object* object, void (Object::*method)(ArgsF...), std::tuple<ArgsT...> const& argsTuple) {
    internal::TupleUnwrapper<sizeof...(ArgsT)>::applyTuple(object, method, argsTuple);
}


/**
 * @brief Callable message having the operator () as payload.
 */
class Callable {
public:
    virtual void operator()() = 0;
};


/**
 * @brief Callable message wrapping a call to a class method
 * @tparam Object   The object class
 * @tparam Args     The arguments to pass
 */
template<class Object, typename ...Args>
class MethodCaller : public Callable {
    using MethodPtr = void (Object::*)(Args...);
private:
    Object* object;             //!< pointer to the instance
    MethodPtr method;           //!< pointer to the method to call
    std::tuple<Args...> args;   //!< a tuple containing arguments to be passed to the method

public:
    /**
     * @brief Construct a new method caller
     * @param object    Pointer to the object
     * @param method    Pointer to the class method to be called on the object
     * @param args      Arguments to pass to the object method
     */
    MethodCaller(Object* object, MethodPtr method, Args... args):
        object(object), method(method), args(args...)
    {}

    virtual void operator()() {
        applyTuple(object, method, args);
    }
};


/**
 * @brief Isolated thread executing callable messages.
 * Provides call(..) method registering a class method to be called in the isolated thread on a given set of arguments
 * and blocking until the call is actually performed. Possible exceptions occurred during the call are reported back to
 * the calling thread.
 */
class IsolatedThread {
    private:
        bool isRunning;                                 //!< while `true`, the managing thread is kept alive
        std::mutex access;                              //!< access control to all the shared resources
        std::condition_variable threadNotifier;         //!< condition variable to notify the thread about new messages
        std::condition_variable callerNotifier;         //!< condition variable to notify the callers about processed messagess 
        std::deque<Callable*> messages;                 //!< callable messages queue
        std::map<int, std::exception_ptr> exceptions;   //!< maps message numbers to exceptions thrown during the processing
        std::thread thread;                             //!< thread executing the callable messages
        int submittedMsgNumber;                         //!< number of messages submitted to the queue
        int processedMsgNumber;                         //!< number of messages processed by the managing thread

        /**
         * @brief Thread function consuming and executing the callable messages
         */
        void threadFunc();

    public:
        IsolatedThread();
        ~IsolatedThread();

        /**
         * @brief Calls something in a Callable in the thread
         * @param entry the callable message pointer. It is destroyed internally after the call.
         */
        void call(Callable* entry);

        /**
         * @brief Calls a given method of an object in the thread
         * @param object        Pointer to the object
         * @param function      Pointer to the class method to be called on the object
         * @param args          Arguments to pass to the object method
         */
        template<class Object, typename ...Args, typename ...Args_>
        void call(Object* object, void (Object::*function)(Args...), Args_&... args) {
            call(new MethodCaller<Object, Args...>(object, function, args...));
        }
};

}  // namespace upstride