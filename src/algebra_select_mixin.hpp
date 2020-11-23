/**
 * @file algebra_select_mixin.hpp
 * @author Maxim Karpushin (maxim.karpushin@upstride.io)
 * @brief Utility class invoking specialized compute code according to an algebra known at runtime
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <stdexcept>

#include "algebras.hpp"

namespace upstride {

/**
 * @brief Selects a specialized implementation of a derived class method according to a runtime algebra value
 * 
 * @tparam Base The derived class type
 */
template <class Base>
class AlgebraSelectionMixin {
   protected:
    /**
     * @brief Calls proceedWithAlgebra() method of a derived class specialized according to a runtime-known algebra
     * 
     * @tparam Args     template parameter pack
     * @param algebra   Algebra value
     * @param args      Arguments to pass to the specialized function
     */
    template <typename... Args>
    void proceedWithAlgebra(Algebra algebra, Args&&... args) {
        switch (algebra) {
            case Algebra::REAL:
                static_cast<Base*>(this)->template proceedWithAlgebra<Algebra::REAL>(args...);
                return;

            case Algebra::COMPLEX:
                static_cast<Base*>(this)->template proceedWithAlgebra<Algebra::COMPLEX>(args...);
                return;

            case Algebra::QUATERNION:
                static_cast<Base*>(this)->template proceedWithAlgebra<Algebra::QUATERNION>(args...);
                return;

            case Algebra::GA_300:
                static_cast<Base*>(this)->template proceedWithAlgebra<Algebra::GA_300>(args...);
                return;
        }

        throw std::runtime_error("Invalid algebra");
    }
};

}  // namespace upstride