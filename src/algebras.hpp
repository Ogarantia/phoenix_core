/**
 * @file algebras.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Geometric algebras declaration
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once

namespace upstride {

/**
 * @brief Algebra specification
 */
enum Algebra {
    REAL,
    QUATERNION
};

/**
 * @brief Clifford product sign table entry
 */
typedef struct {
    int left, right;  //!< numbers of left and right components
    bool positive;    //!< if `true`, the contribution of the multiplication the two given components is positive (negative otherwise)
} SignTableEntry;

/**
 * @brief Row of sign tables in Clifford product specification
 */
typedef struct {
    const SignTableEntry elements[];
} SignTableRow;

/**
 * @brief Forward declaration of Clifford product specification
 * @tparam algebra  Algebra specification
 */
template <int algebra>
class CliffordProductSpec;

template <>
class CliffordProductSpec<Algebra::REAL> {
   public:
    static const int DIMS = 1;
    static const SignTableRow SIGNTABLE[];
};

template <>
class CliffordProductSpec<Algebra::QUATERNION> {
   public:
    static const int DIMS = 4;
    static const SignTableRow SIGNTABLE[];
};

}  // namespace upstride