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
 * @brief Forward declaration of Clifford product specification
 * @tparam algebra  Algebra specification
 */
template <int algebra>
class CliffordProductSpec;

template <>
class CliffordProductSpec<Algebra::REAL> {
   public:
    static const int DIMS = 1;
    static const SignTableEntry SIGNTABLE[];    //!< specifies the contribution of every left-right component pair to the product
    static const int SIGNTABLE_LAYOUT[];        //!< index of the first entry of every row in the signtable
};

template <>
class CliffordProductSpec<Algebra::QUATERNION> {
   public:
    static const int DIMS = 4;
    static const SignTableEntry SIGNTABLE[];    //!< specifies the contribution of every left-right component pair to the product
    static const int SIGNTABLE_LAYOUT[];        //!< index of the first entry of every row in the signtable
};

}  // namespace upstride