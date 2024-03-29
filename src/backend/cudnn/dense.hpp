/**
 * @file dense.hpp
 * @author Philipe Moura (philipe.moura@upstride.io)
 * @brief dense implementation using cuBLAS compute backend
 *
 * @copyright Copyright (c) 2020 UpStride
 */

#pragma once
#include <cublas_v2.h>
#include "../backend.hpp"
#include "context.hpp"
#include "kernels.hpp"

namespace upstride {
    namespace cublas {
        /**
         * @brief Wraps cuBLAS default GEMM
         * @tparam T        Datatype of matrix entries
         * @param device    GPU device the computations are run on
         * @param transa    Conjugation flag for the left matrix
         * @param transb    Conjugation flag for the right matrix
         * @param m         Number of rows in the left matrix
         * @param n         Number of columns in the right matrix
         * @param k         Number of columns in the left matrix = number of rows in the right matrix
         * @param A         Pointer to the left matrix entries
         * @param lda       Leading dimension of two-dimensional array used to store the left matrix
         * @param B         Pointer to the right matrix entries
         * @param ldb       Leading dimension of two-dimensional array used to store the right matrix
         * @param C         Pointer to the output matrix
         * @param ldc       Leading dimension of two-dimensional array used to store the output matrix
         */
        template <typename T>
        inline void gemm(
            const device::CUDA& device, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T* A, int lda, const T* B,
            int ldb, T* C, int ldc);

        template <>
        inline void gemm(
            const device::CUDA& device, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* A, int lda, const float* B,
            int ldb, float* C, int ldc)
        {
            const float zero = 0.0f, one = 1.0f;
            cudnn::Context::raiseIfError(cublasSgemm(device.getCublasHandle(), transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc));
        }

#ifdef UPSTRIDE_ENABLE_FP16
        template <>
        inline void gemm(
            const device::CUDA& device, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* A, int lda, const half* B,
            int ldb, half* C, int ldc)
        {
            // Mimicking TensorFlow here: using 32 bits accumulation for better precision
            const float zero(0.0f), one(1.0f);
            cudnn::Context::raiseIfError(cublasSgemmEx(device.getCublasHandle(), transa, transb, m, n, k, &one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &zero, C, CUDA_R_16F, ldc));
        }
#endif
    }


    /**
     * @brief Dense implementation using cuBLAS.
     * @tparam T    A scalar datatype for the tensor content
     */
    template <typename T>
    class ScalarDenseFunctor<device::CUDA, T> {
    private:
        cudnn::Context& context;
        const FilterLayout weightsLayout;

    public:
        /**
         * @brief Sets main dense parameters independent from the input, filter and output sizes
         * @param context             A context instance
         * @param device              A device instance the operation is performed on
         * @param weightsLayout       Weights matrix layout
         * @param inputShape          Input tensor shape
         * @param kernelShape         kernel tensor shape
         * @param biasShape           Bias tensor shape; empty if the bias addition is disabled
         * @param outputShape         Output tensor shape
         */
        ScalarDenseFunctor(
            upstride::Context &context,
            device::CUDA& device,
            FilterLayout weightsLayout,
            const Shape &inputShape,
            const Shape &kernelShape,
            const Shape &biasShape,
            const Shape &outputShape
        ):
            context(static_cast<cudnn::Context&>(context)),
            weightsLayout(weightsLayout)
        { }

        /**
         * @brief Executes the convolution operation
         * @param inputTensor       Input tensor
         * @param filterTensor      Filter tensor
         * @param biasTensor        Pointer to bias tensor; may be null
         * @param outputTensor      Output tensor
         */
        void operator()(const Tensor<device::CUDA, const T> &inputTensor,
                        const Tensor<device::CUDA, const T> &filterTensor,
                        const Tensor<device::CUDA, const T> *biasTensor,
                        Tensor<device::CUDA, T> &outputTensor)
        {
            // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
            int m = inputTensor.getShape()[0];
            int n = filterTensor.getShape()[1];
            int k = inputTensor.getShape()[1];

            // cublas*gemm uses column-major order. More explanation in the links below:
            // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
            // https://en.wikipedia.org/wiki/Row-_and_column-major_order
            // If weight is received transposed
            if (this->weightsLayout == upstride::FilterLayout::OI) {
                cublas::gemm(
                    inputTensor.getDevice(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                    filterTensor.getDataPtr(), k,
                    inputTensor.getDataPtr(), k,
                    outputTensor.getDataPtr(), n);
            } else {
                cublas::gemm(
                    inputTensor.getDevice(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    filterTensor.getDataPtr(), n,
                    inputTensor.getDataPtr(), k,
                    outputTensor.getDataPtr(), n);
            }
            // add bias
            if (biasTensor) {
                cudnn::addBias(outputTensor, *biasTensor, upstride::DataFormat::NC);
            }
        }
    };


    /**
     * @brief dense backward implementation using cuBLAS
     * @tparam T    scalar datatype
     */
    template <typename T>
    class ScalarDenseGradFunctor<device::CUDA, T> {
    private:
        cudnn::Context& context;
        const FilterLayout weightsLayout;
        const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not

    public:
        /**
         * @brief Instantiates dense layer gradient operation
         * @param context                 A context instance
         * @param device                  A device instance the operation is performed on
         * @param weightsLayout           Weights matrix layout
         * @param inputShape              Input tensor shape
         * @param kernelShape             kernel tensor shape
         * @param outputShape             Output tensor shape
         * @param requireInputGrad        If `true`, the computation of the gradient w.r.t the input is enabled.
         */
        ScalarDenseGradFunctor(
            upstride::Context& context,
            device::CUDA& device,
            FilterLayout weightsLayout,
            const Shape &inputShape,
            const Shape &kernelShape,
            const Shape &outputShape,
            bool requireInputGrad
        ):
            context(static_cast<cudnn::Context&>(context)),
            weightsLayout(weightsLayout),
            requireInputGrad(requireInputGrad)
        { }

        /**
        * @brief Executes the operation
        * @param inputTensor       forward input tensor
        * @param kernelTensor      forward input kernel tensor
        * @param gradTensor        gradient of the forward output tensor (dy)
        * @param kernelGradTensor  output: kernel gradient
        * @param inputGradTensor   output: input gradient
        */
        void operator()(const Tensor<device::CUDA, const T>& inputTensor,
                        const Tensor<device::CUDA, const T>& kernelTensor,
                        const Tensor<device::CUDA, const T>& gradTensor,
                        Tensor<device::CUDA, T>& kernelGradTensor,
                        Tensor<device::CUDA, T>& inputGradTensor) {

            int m = inputTensor.getShape()[0];
            int n = gradTensor.getShape()[1];
            int k = inputTensor.getShape()[1];

            cublas::gemm(
                inputTensor.getDevice(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                gradTensor.getDataPtr(), n,
                inputTensor.getDataPtr(), k,
                kernelGradTensor.getDataPtr(), n);

            if (requireInputGrad) {
                m = gradTensor.getShape()[0];
                n = kernelTensor.getShape()[0];
                k = gradTensor.getShape()[1];

                // If weight is received transposed
                if (this->weightsLayout == upstride::FilterLayout::OI) {
                    cublas::gemm(
                        gradTensor.getDevice(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                        kernelTensor.getDataPtr(), n,
                        gradTensor.getDataPtr(), k,
                        inputGradTensor.getDataPtr(), n);
                } else {
                    cublas::gemm(
                        gradTensor.getDevice(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                        kernelTensor.getDataPtr(), k,
                        gradTensor.getDataPtr(), k,
                        inputGradTensor.getDataPtr(), n);
                }
            }
        }
    };

} // namespace upstride