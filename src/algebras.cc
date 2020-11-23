#include "algebras.hpp"

using namespace upstride;

const SignTableEntry CliffordProductSpec<Algebra::REAL>::SIGNTABLE[] = {{0, 0, true}};
const int CliffordProductSpec<Algebra::REAL>::SIGNTABLE_LAYOUT[] = {0, 1};
const int CliffordProductSpec<Algebra::REAL>::BACKPROP_ORDER[] = {0};

//
// COMPLEX ALGEBRA
//
const SignTableEntry CliffordProductSpec<Algebra::COMPLEX>::SIGNTABLE[] = {
    // real part
    {0, 0, true},   // r*r
    {1, 1, false},  // -i*i
    // imaginary part
    {0, 1, true},   // r*i
    {1, 0, true}    // i*r
};

const int CliffordProductSpec<Algebra::COMPLEX>::SIGNTABLE_LAYOUT[] = {0, 2, 4};

const int CliffordProductSpec<Algebra::COMPLEX>::BACKPROP_ORDER[] = {2, 3, 0, 1};

//
// QUATERNION ALGEBRA
//
const SignTableEntry CliffordProductSpec<Algebra::QUATERNION>::SIGNTABLE[] = {
    // real part
    {0, 0, true},   // r*r
    {1, 1, false},  // -i*i
    {2, 2, false},  // -j*j
    {3, 3, false},  // -k*k
    // i part
    {0, 1, true},   // r*i
    {1, 0, true},   // i*r
    {2, 3, true},   // j*k
    {3, 2, false},  // -k*j
    // j part
    {0, 2, true},   // r*j
    {1, 3, false},  // -i*k
    {2, 0, true},   // j*r
    {3, 1, true},   // k*i
    // k part
    {0, 3, true},   // r*k
    {1, 2, true},   // i*j
    {2, 1, false},  // -j*i
    {3, 0, true}    // k*r
};

const int CliffordProductSpec<Algebra::QUATERNION>::SIGNTABLE_LAYOUT[] = {0, 4, 8, 12, 16};

const int CliffordProductSpec<Algebra::QUATERNION>::BACKPROP_ORDER[] = {10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15};

//
// GA_300
//
const SignTableEntry CliffordProductSpec<Algebra::GA_300>::SIGNTABLE[] = {
    // 0
    { 0, 0, true },
    { 1, 1, true },
    { 2, 2, true },
    { 3, 3, true },
    { 4, 4, false },
    { 5, 5, false },
    { 6, 6, false },
    { 7, 7, false },
    // 1
    { 0, 1, true },
    { 1, 0, true },
    { 2, 4, false },
    { 3, 5, false },
    { 4, 2, true },
    { 5, 3, true },
    { 6, 7, false },
    { 7, 6, false },
    // 2
    { 0, 2, true },
    { 1, 4, true },
    { 2, 0, true },
    { 3, 6, false },
    { 4, 1, false },
    { 5, 7, true },
    { 6, 3, true },
    { 7, 5, true },
    // 3
    { 0, 3, true },
    { 1, 5, true },
    { 2, 6, true },
    { 3, 0, true },
    { 4, 7, false },
    { 5, 1, false },
    { 6, 2, false },
    { 7, 4, false },
    // 4
    { 0, 4, true },
    { 1, 2, true },
    { 2, 1, false },
    { 3, 7, true },
    { 4, 0, true },
    { 5, 6, false },
    { 6, 5, true },
    { 7, 3, true },
    // 5
    { 0, 5, true },
    { 1, 3, true },
    { 2, 7, false },
    { 3, 1, false },
    { 4, 6, true },
    { 5, 0, true },
    { 6, 4, false },
    { 7, 2, false },
    // 6
    { 0, 6, true },
    { 1, 7, true },
    { 2, 3, true },
    { 3, 2, false },
    { 4, 5, false },
    { 5, 4, true },
    { 6, 0, true },
    { 7, 1, true },
    // 7
    { 0, 7, true },
    { 1, 6, true },
    { 2, 5, false },
    { 3, 4, true },
    { 4, 3, true },
    { 5, 2, false },
    { 6, 1, true },
    { 7, 0, true }
};

const int CliffordProductSpec<Algebra::GA_300>::SIGNTABLE_LAYOUT[] = {0, 8, 16, 24, 32, 40, 48, 56, 64};

/**
 * Computed by render_signtable() in clifford_product.py script
 * First 8 terms pick entries form SIGNTABLE that contribute positively to the output and cover all the 8 left
 * and right components. The remaining terms are put in the default order.
 */
const int CliffordProductSpec<Algebra::GA_300>::BACKPROP_ORDER[] = {
    0, 1, 2, 21, 22, 23, 44, 59, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63
};