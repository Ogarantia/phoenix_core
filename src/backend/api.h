/**
 * @file api.h
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Backends listing
 * Provides a complete backend translation unit by regroups all available implementations of the backend interface.
 * This file is intended to be included in the core level, in order to provide the core operations implementations with the properly defined computational routines.
 * @copyright Copyright (c) 2020 UpStride
 */

// oneDNN CPU backend implementation
#include "onednn/tensor.hpp"
#include "onednn/conv2d.hpp"
#include "onednn/dense.hpp"

// cuDNN GPU backend implementation
#ifdef BACKEND_CUDNN
#include "cudnn/half.hpp"
#include "cudnn/tensor.hpp"
#include "cudnn/conv2d.hpp"
#include "cudnn/dense.hpp"
#include "cudnn/quat_pointwise_conv2d.hpp"
#endif
