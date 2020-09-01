#pragma once
#include <assert.h>

#include "../backend.hpp"
#include "context.hpp"

namespace upstride {

/**
 * @brief Conv2D kernel memory layout required by oneDNN
 */
static const dnnl::memory::format_tag KERNEL_MEMORY_LAYOUT = dnnl::memory::format_tag::oihw;
static const dnnl::memory::format_tag KERNEL_MEMORY_LAYOUT_DW = dnnl::memory::format_tag::goihw;
static const dnnl::memory::format_tag BIAS_MEMORY_LAYOUT = dnnl::memory::format_tag::x;

/**
 * @brief 2D convolution implementation using oneDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class ScalarConv2DFunctor<device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, kernelMemDesc, biasMemDesc, outputMemDesc;
    dnnl::convolution_forward convPrim, convPrimNoBias;
    const dnnl::memory::format_tag formatTag;
    const IntPair stride, dilation;
    const bool useBias;  //!< if `true`, a bias tensor is added to the convolution output

    Shape inputShape, kernelShape, biasShape, outputShape;
    IntPair padBefore;  //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;   //!< zero padding: number of zeros to add at the end to every input spatial dimension

   public:
    /**
     * @brief Sets main convolution parameters independent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     * @param useBias       If `true`, the bias addition is enabled.
     */
    ScalarConv2DFunctor(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool useBias) : formatTag(onednn::dataFormatToFormatTag(dataFormat)),
                                                                                                               stride(stride),
                                                                                                               dilation(dilation),
                                                                                                               useBias(useBias) {}

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param kernelShape       kernel tensor shape
     * @param biasShape         Bias tensor shape; may be empty if the bias addition is not enabled by `useBias`
     * @param outputTensor      Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void configure(const Shape& inputShape,
                   const Shape& kernelShape,
                   const Shape& biasShape,
                   const Shape& outputShape,
                   const IntPair& padBefore,
                   const IntPair& padAfter,
                   const int groups = 1) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->kernelShape == kernelShape &&
            (!useBias || this->biasShape == biasShape) && this->outputShape == outputShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->kernelShape = kernelShape;
        if (useBias)
            this->biasShape = biasShape;
        this->outputShape = outputShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // set up oneDNN memory descriptors
        inputMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), formatTag);
        if (groups == 1) {
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT);
        } else {
            // converting OIHW shape into GOIHW
            Shape kernelShapeExpanded = kernelShape.expandDim(0);
            kernelShapeExpanded[0] = groups;
            kernelShapeExpanded[1] /= groups;
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShapeExpanded), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT_DW);
        }

        // bias vector must be of the output channel size, otherwise that mean that we don't use a bias
        if (useBias)
            biasMemDesc = dnnl::memory::desc(dnnl::memory::dims{biasShape.numel()}, onednn::getDataType<T>(), BIAS_MEMORY_LAYOUT);

        outputMemDesc = dnnl::memory::desc(onednn::shapeToDims(outputShape), onednn::getDataType<T>(), formatTag);

        // set up convolution operation-related descriptors
        if (useBias) {
            // biased convolution
            convPrim = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
                dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                                dnnl::algorithm::convolution_auto,
                                                inputMemDesc, kernelMemDesc, biasMemDesc, outputMemDesc,
                                                dnnl::memory::dims({stride.y, stride.x}),
                                                dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                                                dnnl::memory::dims({padBefore.y, padBefore.x}),
                                                dnnl::memory::dims({padAfter.y, padAfter.x})),
                onednn::Context::getInstance().getEngine()));
        }

        // biasless convolution (it is setup anyway to be able to use both biased and biasless versions)
        convPrimNoBias = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, kernelMemDesc, outputMemDesc,
                                            dnnl::memory::dims({stride.y, stride.x}),
                                            dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                                            dnnl::memory::dims({padBefore.y, padBefore.x}),
                                            dnnl::memory::dims({padAfter.y, padAfter.x})),
            onednn::Context::getInstance().getEngine()));
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param kernelTensor      Kernel tensor
     * @param biasTensor        Pointer to bias tensor; may be null
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<device::CPU, const T>& inputTensor,
                    const Tensor<device::CPU, const T>& kernelTensor,
                    const Tensor<device::CPU, const T>* biasTensor,
                    Tensor<device::CPU, T>& outputTensor) {
        // instantiate DNNL memory
        auto& engine = onednn::Context::getInstance().getEngine();
        dnnl::memory input(inputMemDesc, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory kernel(kernelMemDesc, engine, const_cast<T*>(kernelTensor.getDataPtr()));
        dnnl::memory output(outputMemDesc, engine, outputTensor.getDataPtr());

        if (biasTensor) {
            if (!useBias)
                throw std::invalid_argument("Bias application is not configured for this scalar Conv2D operation");

            dnnl::memory bias(biasMemDesc, engine, const_cast<T*>(biasTensor->getDataPtr()));
            onednn::Context::getInstance().execute(convPrim, {{DNNL_ARG_SRC, input},
                                                              {DNNL_ARG_WEIGHTS, kernel},
                                                              {DNNL_ARG_BIAS, bias},
                                                              {DNNL_ARG_DST, output}});
        } else {
            onednn::Context::getInstance().execute(convPrimNoBias, {{DNNL_ARG_SRC, input},
                                                                    {DNNL_ARG_WEIGHTS, kernel},
                                                                    {DNNL_ARG_DST, output}});
        }
    }
};

/**
 * @brief 2D backward convolution implementation using oneDNN
 * @tparam T    scalar datatype 
 */
template <typename T>
class ScalarConv2DGradFunctor<device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, kernelMemDesc, gradMemDesc, kernelGradMemDesc, inputGradMemDesc;
    dnnl::convolution_backward_data convBackDataPrim;
    dnnl::convolution_backward_weights convBackWeightsPrim;
    const dnnl::memory::format_tag formatTag;
    const IntPair stride, dilation;
    const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not
    Shape inputShape, kernelShape, gradShape;
    IntPair padBefore;  //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;   //!< zero padding: number of zeros to add at the end to every input spatial dimension

   public:
    ScalarConv2DGradFunctor(DataFormat dataFormat, const IntPair& stride, const IntPair& dilation, bool requireInputGrad) : formatTag(onednn::dataFormatToFormatTag(dataFormat)),
                                                                                                                            stride(stride),
                                                                                                                            dilation(dilation),
                                                                                                                            requireInputGrad(requireInputGrad) {}

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param kernelShape       kernel tensor shape
     * @param gradShape         grad tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void configure(const Shape& inputShape,
                   const Shape& kernelShape,
                   const Shape& gradShape,
                   const IntPair& padBefore,
                   const IntPair& padAfter,
                   const int groups = 1) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->kernelShape == kernelShape && this->gradShape == gradShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->kernelShape = kernelShape;
        this->gradShape = gradShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // set up oneDNN memory descriptors
        // for inputs
        inputMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), formatTag);
        if (groups == 1) {
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT);
        } else {
            // converting OIHW shape into GOIHW
            Shape kernelShapeExpanded = kernelShape.expandDim(0);
            kernelShapeExpanded[0] = groups;
            kernelShapeExpanded[1] /= groups;
            kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShapeExpanded), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT_DW);
        }
        gradMemDesc = dnnl::memory::desc(onednn::shapeToDims(gradShape), onednn::getDataType<T>(), formatTag);
        // for output
        kernelGradMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT);
        inputGradMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), formatTag);

        // Re-computation of the convolution forward primitive descriptor
        dnnl::convolution_forward::primitive_desc convPd(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, kernelMemDesc, gradMemDesc,
                                            dnnl::memory::dims({stride.y, stride.x}),
                                            dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                                            dnnl::memory::dims({padBefore.y, padBefore.x}),
                                            dnnl::memory::dims({padAfter.y, padAfter.x})),
            onednn::Context::getInstance().getEngine());

        // instantiate backward conv primitive to compute the kernel gradient
        convBackWeightsPrim = dnnl::convolution_backward_weights(
            dnnl::convolution_backward_weights::primitive_desc(
                dnnl::convolution_backward_weights::desc(
                    dnnl::algorithm::convolution_auto,
                    inputMemDesc,   // conv_diff_dst_md
                    kernelMemDesc,  // conv_diff_weights_md
                    gradMemDesc,    //  conv_bwd_src_md
                    dnnl::memory::dims({stride.y, stride.x}),
                    dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                    dnnl::memory::dims({padBefore.y, padBefore.x}),
                    dnnl::memory::dims({padAfter.y, padAfter.x})),
                onednn::Context::getInstance().getEngine(),
                convPd));

        // instantiate backward conv primitive to compute the input gradient (only if required)
        if (requireInputGrad) {
            convBackDataPrim = dnnl::convolution_backward_data(
                dnnl::convolution_backward_data::primitive_desc(
                    dnnl::convolution_backward_data::desc(
                        dnnl::algorithm::convolution_direct,
                        inputMemDesc,   // conv_diff_dst_md
                        kernelMemDesc,  // conv_diff_weights_md
                        gradMemDesc,    //  conv_bwd_src_md
                        dnnl::memory::dims({stride.y, stride.x}),
                        dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                        dnnl::memory::dims({padBefore.y, padBefore.x}),
                        dnnl::memory::dims({padAfter.y, padAfter.x})),
                    onednn::Context::getInstance().getEngine(),
                    convPd));
        }
    }

    /**
     * @brief Executes the convolution operation
     * 
     * @param inputTensor       forward input tensor
     * @param kernelTensor      forward input kernel tensor
     * @param gradTensor        gradient of the forward output tensor (dy)
     * @param kernelGradTensor  output: kernel gradient
     * @param inputGradTensor   output: input gradient
     * @param padBefore         number of zero samples to add to the input tensor on top/left
     * @param padAfter          number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups in order to manage groups convolutions and mostly the depthwise convolution (groups == Input channels), 1 by default (regular convolution)
     */
    void operator()(const Tensor<device::CPU, const T>& inputTensor,
                    const Tensor<device::CPU, const T>& kernelTensor,
                    const Tensor<device::CPU, const T>& gradTensor,
                    Tensor<device::CPU, T>& kernelGradTensor,
                    Tensor<device::CPU, T>& inputGradTensor) {
        // instantiate DNNL memory
        auto& engine = onednn::Context::getInstance().getEngine();
        dnnl::memory input(inputMemDesc, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory kernel(kernelMemDesc, engine, const_cast<T*>(kernelTensor.getDataPtr()));
        dnnl::memory grad(gradMemDesc, engine, const_cast<T*>(gradTensor.getDataPtr()));

        dnnl::memory kernelGrad(kernelGradMemDesc, engine, kernelGradTensor.getDataPtr());
        dnnl::memory inputGrad(inputGradMemDesc, engine, inputGradTensor.getDataPtr());

        // g-g-go
        onednn::Context::getInstance().execute(convBackWeightsPrim, {{DNNL_ARG_SRC, input},
                                                                     {DNNL_ARG_DIFF_DST, grad},
                                                                     {DNNL_ARG_DIFF_WEIGHTS, kernelGrad}}  // output
        );
        if (requireInputGrad) {
            onednn::Context::getInstance().execute(convBackDataPrim, {{DNNL_ARG_WEIGHTS, kernel},
                                                                      {DNNL_ARG_DIFF_DST, grad},
                                                                      {DNNL_ARG_DIFF_SRC, inputGrad}}  //output
            );
        }
    }
};

}  // namespace upstride