#pragma once

#include "../backend.hpp"
#include "context.hpp"

namespace upstride
{

    /**
     * @brief Dense kernel memory layout required by oneDNN
     */
    static const dnnl::memory::format_tag PLAIN_2D_TENSOR_MEMORY_LAYOUT = dnnl::memory::format_tag::nc;
    static const dnnl::memory::format_tag TRANSPOSED_2D_TENSOR_MEMORY_LAYOUT = dnnl::memory::format_tag::cn;
    static const dnnl::memory::format_tag BIAS_DENSE_MEMORY_LAYOUT = dnnl::memory::format_tag::nc;

    /**
     * @brief dense implementation using oneDNN
     * @tparam T    scalar datatype
     */
    template <typename T>
    class ScalarDenseFunctor<device::CPU, T>
    {
    private:
        onednn::Context& context;
        dnnl::memory::desc inputMemDesc, kernelMemDesc, biasMemDesc, outputMemDesc;
        dnnl::matmul densePrim, densePrimNoBias;
        const dnnl::memory::format_tag formatTag;
        const bool useBias; //!< if `true`, a bias tensor is added to the dense output

        Shape inputShape, kernelShape, biasShape, outputShape;

    public:
        /**
         * @brief Sets main dense parameters independent from the input, filter and output sizes
         * @param context       A context instance
         * @param dataFormat    Expected tensors format
         * @param useBias       If `true`, the bias addition is enabled.
         */
        ScalarDenseFunctor(upstride::Context& context, DataFormat dataFormat, bool useBias) : context(static_cast<onednn::Context&>(context)),
                                                                                              formatTag(onednn::dataFormatToFormatTag(dataFormat)),
                                                                                              useBias(useBias) {}

        /**
         * @brief Performs backend-related operation configuration
         * @param inputShape        Input tensor shape
         * @param kernelShape       kernel tensor shape
         * @param biasShape         Bias tensor shape; may be empty if the bias addition is not enabled by `useBias`
         * @param outputTensor      Output tensor shape
         */
        void configure(const Shape &inputShape,
                       const Shape &kernelShape,
                       const Shape &biasShape,
                       const Shape &outputShape)
        {
            // check if up-to-date
            if (this->inputShape == inputShape && this->kernelShape == kernelShape &&
                (!useBias || this->biasShape == biasShape) && this->outputShape == outputShape)
                return;

            // cache shapes for further up-to-dateness checks
            this->inputShape = inputShape;
            this->kernelShape = kernelShape;
            if (useBias)
                this->biasShape = biasShape;
            this->outputShape = outputShape;

            // set up oneDNN memory descriptors
            inputMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), PLAIN_2D_TENSOR_MEMORY_LAYOUT);
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), formatTag);

            // bias vector must be of the output channel size, otherwise that mean that we don't use a bias
            if (useBias) {
                biasMemDesc = dnnl::memory::desc(onednn::shapeToDims(biasShape), onednn::getDataType<T>(), BIAS_DENSE_MEMORY_LAYOUT);
            }

            outputMemDesc = dnnl::memory::desc(onednn::shapeToDims(outputShape), onednn::getDataType<T>(), formatTag);

            // set up dense operation-related descriptors
            if (useBias)
            {
                // biased dense
                dnnl::matmul::desc matmulDescriptor(inputMemDesc, kernelMemDesc, biasMemDesc, outputMemDesc);
                densePrim = dnnl::matmul(dnnl::matmul::primitive_desc(matmulDescriptor, context.getEngine()));
            }

            // biasless dense (it is setup anyway to be able to use both biased and biasless versions)
            dnnl::matmul::desc matmulDescriptorNoBias(inputMemDesc, kernelMemDesc, outputMemDesc);
            densePrimNoBias = dnnl::matmul(dnnl::matmul::primitive_desc(matmulDescriptorNoBias, context.getEngine()));
        }

        /**
         * @brief Executes the dense operation
         * @param inputTensor       Input tensor
         * @param kernelTensor      Kernel tensor
         * @param biasTensor        Pointer to bias tensor; may be null
         * @param outputTensor      Output tensor
         */
        void operator()(const Tensor<device::CPU, const T> &inputTensor,
                        const Tensor<device::CPU, const T> &kernelTensor,
                        const Tensor<device::CPU, const T> *biasTensor,
                        Tensor<device::CPU, T> &outputTensor)
        {
            // instantiate DNNL memory
            auto &engine = context.getEngine();
            dnnl::memory input(inputMemDesc, engine, const_cast<T *>(inputTensor.getDataPtr()));
            dnnl::memory kernel(kernelMemDesc, engine, const_cast<T *>(kernelTensor.getDataPtr()));
            dnnl::memory output(outputMemDesc, engine, outputTensor.getDataPtr());

            if (biasTensor) {
                if (!useBias)
                    throw std::invalid_argument("Bias application is not configured for this scalar Dense operation");

                dnnl::memory bias(biasMemDesc, engine, const_cast<T *>(biasTensor->getDataPtr()));
                context.execute(densePrim, {{DNNL_ARG_SRC, input},
                                                                   {DNNL_ARG_WEIGHTS, kernel},
                                                                   {DNNL_ARG_BIAS, bias},
                                                                   {DNNL_ARG_DST, output}});
            }
            else {
                context.execute(densePrimNoBias, {{DNNL_ARG_SRC, input},
                                                                         {DNNL_ARG_WEIGHTS, kernel},
                                                                         {DNNL_ARG_DST, output}});
            }
        }
    };



    /**
     * @brief backward dense implementation using oneDNN
     * @tparam T    scalar datatype
     */
    template <typename T>
    class ScalarDenseGradFunctor<device::CPU, T> {
    private:
        onednn::Context& context;
        dnnl::memory::desc inputMemDescTransposed, kernelMemDescTransposed, gradMemDesc, kernelGradMemDesc, inputGradMemDesc;
        dnnl::matmul denseBackDataPrim, denseBackWeightPrim;
        const dnnl::memory::format_tag formatTag;
        const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not
        Shape inputShape, kernelShape, gradShape;

    public:
        /**
         * @brief Sets main dense parameters independent from the input, filter and output sizes
         * @param context           A context instance
         * @param dataFormat        Expected tensors format
         * @param requireInputGrad  If `true`, the computation of the gradient w.r.t the input is enabled.
         */
        ScalarDenseGradFunctor(upstride::Context& context, DataFormat dataFormat, bool requireInputGrad) : 
                                                                                    context(static_cast<onednn::Context&>(context)),
                                                                                    formatTag(onednn::dataFormatToFormatTag(dataFormat)),
                                                                                    requireInputGrad(requireInputGrad) { }

        /**
         * @brief Performs backend-related operation configuration
         * @param inputShape        Input tensor shape
         * @param kernelShape       kernel tensor shape
         * @param gradShape         grad tensor shape
         */
        void configure(const Shape& inputShape,
                    const Shape& kernelShape,
                    const Shape& gradShape) {
            // check if up-to-date
            if (this->inputShape == inputShape && this->kernelShape == kernelShape && this->gradShape == gradShape)
                return;

            // cache shapes for further up-to-dateness checks
            this->inputShape = inputShape;
            this->kernelShape = kernelShape;
            this->gradShape = gradShape;

            // set up oneDNN memory descriptors
            // for inputs
            dnnl::memory::desc inputMemDesc, kernelMemDesc, gradMemDesc;
            inputMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), PLAIN_2D_TENSOR_MEMORY_LAYOUT);
            inputMemDescTransposed = inputMemDesc.permute_axes({1, 0});
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), formatTag);
            kernelMemDescTransposed = kernelMemDesc.permute_axes({1, 0});
            gradMemDesc = dnnl::memory::desc(onednn::shapeToDims(gradShape), onednn::getDataType<T>(), PLAIN_2D_TENSOR_MEMORY_LAYOUT);
            // for output
            kernelGradMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), PLAIN_2D_TENSOR_MEMORY_LAYOUT);
            inputGradMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), PLAIN_2D_TENSOR_MEMORY_LAYOUT);

            dnnl::matmul::desc matmulDescriptorDWeight(inputMemDescTransposed, gradMemDesc, kernelGradMemDesc);
            denseBackWeightPrim = dnnl::matmul(dnnl::matmul::primitive_desc(matmulDescriptorDWeight, context.getEngine()));

            // instantiate backward dense primitive to compute the input gradient (only if required)
            if (requireInputGrad) {
                dnnl::matmul::desc matmulDescriptorDInput(gradMemDesc, kernelMemDescTransposed, inputGradMemDesc);
                denseBackDataPrim = dnnl::matmul(dnnl::matmul::primitive_desc(matmulDescriptorDInput, context.getEngine()));
            }
        }

        /**
         * @brief Executes the dense operation
         * @param inputTensor       forward input tensor
         * @param kernelTensor      forward input kernel tensor
         * @param gradTensor        gradient of the forward output tensor (dy)
         * @param kernelGradTensor  output: kernel gradient
         * @param inputGradTensor   output: input gradient
         */
        void operator()(const Tensor<device::CPU, const T>& inputTensor,
                        const Tensor<device::CPU, const T>& kernelTensor,
                        const Tensor<device::CPU, const T>& gradTensor,
                        Tensor<device::CPU, T>& kernelGradTensor,
                        Tensor<device::CPU, T>& inputGradTensor) {
            // instantiate DNNL memory
            auto& engine = context.getEngine();
            dnnl::memory inputTransposed(inputMemDescTransposed, engine, const_cast<T*>(inputTensor.getDataPtr()));
            dnnl::memory kernelTransposed(kernelMemDescTransposed, engine, const_cast<T*>(kernelTensor.getDataPtr()));
            dnnl::memory grad(gradMemDesc, engine, const_cast<T*>(gradTensor.getDataPtr()));

            dnnl::memory kernelGrad(kernelGradMemDesc, engine, kernelGradTensor.getDataPtr());
            dnnl::memory inputGrad(inputGradMemDesc, engine, inputGradTensor.getDataPtr());

            context.execute(denseBackWeightPrim, {{DNNL_ARG_SRC, inputTransposed},
                                                                         {DNNL_ARG_WEIGHTS, grad},
                                                                         {DNNL_ARG_DST, kernelGrad}});
            if (requireInputGrad) {
                context.execute(denseBackDataPrim, {{DNNL_ARG_SRC, grad},
                                                                           {DNNL_ARG_WEIGHTS, kernelTransposed},
                                                                           {DNNL_ARG_DST, inputGrad}});
            }
        }
    };
} // namespace upstride