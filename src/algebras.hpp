/**
 * @file algebras.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Geometric algebras declaration
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <functional>
#include <stdexcept>

namespace upstride {

/**
 * @brief Algebra specification
 */
enum Algebra {
    REAL,
    QUATERNION
};

/**
 * @brief Multivector dimension for a specific algebra
 */
static const int MULTIVECTOR_DIM[] = { 1, 4 };

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
    static const SignTableEntry SIGNTABLE[];  //!< specifies the contribution of every left-right component pair to the product
    static const int SIGNTABLE_LAYOUT[];      //!< index of the first entry of every row in the signtable
};

template <>
class CliffordProductSpec<Algebra::QUATERNION> {
   public:
    static const int DIMS = 4;
    static const SignTableEntry SIGNTABLE[];  //!< specifies the contribution of every left-right component pair to the product
    static const int SIGNTABLE_LAYOUT[];      //!< index of the first entry of every row in the signtable
};

/**
 * @brief Applies a scalar binary operation to matrices of multivectors of a specific algebra using a scalar implementation of the operation by isomorphism.
 * The scalar implementation is accessed using a set of callbacks.
 * Additional temporary buffers are needed to store intermediate computation results.
 * @tparam CliffordProductSpec  Specification of Clifford product
 */
template <typename CliffordProductSpec>
struct BinaryOperation {
    /**
     * @brief Callback function computing the result of a scalar operation taking as arguments specific blades of left and right operands.
     * The result is stored at a specific blade in the output storage.
     * @param lhs  Index of the left operand blade given as the left operand to the scalar operation
     * @param rhs  Index of the right operand blade given as the right operand to the scalar operation
     * @param out  Index of the output blade to fill with the result of the operation
     */
    typedef std::function<void(int lhs, int rhs, int out)> FillOutputFunc;

    /**
     * @brief Callback function computing the result of a scalar operation taking as arguments specific blades of left and right operands.
     * The result is store at a specific buffer.
     * @param lhs  Index of the left operand blade given as the left operand to the scalar operation
     * @param rhs  Index of the right operand blade given as the right operand to the scalar operation
     * @param out  Index of the output blade to fill with the result of the operation
     */
    typedef std::function<void(int lhs, int rhs, int buf)> FillBufferFunc;

    /**
     * @brief Callback function accumulating a specific buffer to the output.
     * The result is store at a specific buffer.
     * @param out       Index of the output blade taken as the accumulator
     * @param buf       Index of the buffer to add to or subtract from the output
     * @param positive  If `true`, the buffer is added to the output, otherwise it is subtracted.
     */
    typedef std::function<void(int out, int buf, bool positive)> AccumulateOutputFunc;

    /**
     * @brief Computes the result of a multiplicative binary operation.
     * 
     * @param fillOutputFunc    Callback function filling a specific blade of the output tensor with the result of the operation
     * @param fillBufferFunc    Callback function filling a buffer tensor with the result of the operation
     * @param accOutputFunc     Callback function accumulating a buffer to the output
     */
    inline static void product(const FillOutputFunc& fillOutputFunc, const FillBufferFunc& fillBufferFunc, const AccumulateOutputFunc& accOutputFunc) {
        // loop through dimensions (blades)
        for (int dim = 0; dim < CliffordProductSpec::DIMS; ++dim) {
            const auto row = &CliffordProductSpec::SIGNTABLE[CliffordProductSpec::SIGNTABLE_LAYOUT[dim]];
            if (!row[0].positive)  // negative first term case is not handled yet
                throw std::runtime_error("Not implemented");

            // compute first term
            fillOutputFunc(row[0].left, row[0].right, dim);

            // compute remaining terms and accumulate the output
            for (int termNum = 1; termNum < CliffordProductSpec::DIMS; ++termNum) {
                const auto& entry = row[termNum];

                // compute convolution for two given components
                fillBufferFunc(entry.left, entry.right, dim);

                // accumulate to the output
                accOutputFunc(dim, 0, entry.positive);
            }
        }
    }
};

}  // namespace upstride