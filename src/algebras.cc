#include "algebras.hpp"

using namespace upstride;

const SignTableEntry CliffordProductSpec<Algebra::REAL>::SIGNTABLE[] = {{0, 0, true}};
const int CliffordProductSpec<Algebra::REAL>::SIGNTABLE_LAYOUT[] = {0};

// TODO: this is not tested yet
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

const int CliffordProductSpec<Algebra::QUATERNION>::SIGNTABLE_LAYOUT[] = {0, 4, 8, 12};