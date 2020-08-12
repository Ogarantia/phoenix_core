/**
 * @file api.h
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Backends listing
 * Provides a complete backend translation unit by regroups all available implementations of the backend interface.
 * This file is intended to be included in the core level, in order to provide the core operations implementations with the properly defined computational routines.
 * @copyright Copyright (c) 2020 UpStride
 */

// oneDNN CPU backend implementation
#include "onednn/conv2d.hpp"

// cuDNN GPU backend implementation
#ifdef BACKEND_CUDNN
#include "cudnn/conv2d.hpp"
#endif