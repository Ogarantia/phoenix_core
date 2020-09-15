#include "backend.hpp"
#include <cstdlib>
#include <cstdarg>

using namespace upstride;

/**
 * @brief Retrieves value of an environment variable.
 * @param variable      The variable
 * @return the variable value as a string.
 */
std::string getEnvVar(const char* variable) {
    const char* value = std::getenv(variable);
    return std::string(value ? value : "");
}


/**
 * @brief Retrieves integer value of an environment variable.
 * If no conversion to integer is possible, returns zero.
 * @param variable      The variable
 * @return the integer variable value or zero if cannot interpret the value as integer.
 */
int getIntegerEnvVar(const char* variable) {
    const char* value = std::getenv(variable);
    return value ? strtol(value, nullptr, 10) : 0;
}


Context::Context():
    envVerbose(getIntegerEnvVar("UPSTRIDE_VERBOSE") > 0),
    envOptimizeMemoryUse(getIntegerEnvVar("UPSTRIDE_MEMORY_OPTIMIZED") > 0)
{
    UPSTRIDE_SAYS(*this, "UpStride engine is speaking! Because verbose mode is enabled. Context created.");
    if (envOptimizeMemoryUse)
        UPSTRIDE_SAYS(*this, "Memory-optimized mode: the engine may run slower but uses less memory.");
}


void Context::verbosePrintf(const char* format, ...) const {
#ifdef UPSTRIDE_ALLOW_VERBOSE
    if (envVerbose) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
#endif
}