#pragma once
#include "context.hpp"

namespace upstride {

/**
 * @brief Conv2D kernel memory layout required by oneDNN
 */
static const dnnl::memory::format_tag KERNEL_MEMORY_LAYOUT = dnnl::memory::format_tag::oihw;

/**
 * @brief Regular 2D convolution implementation using oneDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class UpstrideConv2DFunctor<upstride::device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, filterMemDesc, outputMemDesc;
    dnnl::convolution_forward convPrim;
    dnnl::memory::format_tag formatTag;
    Shape inputShape, filterShape, outputShape;
    IntPair padBefore;  //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;   //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair stride, dilation;

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     */
    void configureBackend(const Shape& inputShape, const Shape& filterShape, const Shape& outputShape, const IntPair& padBefore, const IntPair& padAfter) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->filterShape == filterShape && this->outputShape == outputShape &&
            this->padBefore == padBefore && this->padAfter == padAfter)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->filterShape = filterShape;
        this->outputShape = outputShape;
        this->padBefore = padBefore;
        this->padAfter = padAfter;

        // set up oneDNN memory descriptors
        inputMemDesc = dnnl::memory::desc(onednn::shapeToDims(inputShape), onednn::getDataType<T>(), formatTag);
        filterMemDesc = dnnl::memory::desc(onednn::shapeToDims(filterShape), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT);
        outputMemDesc = dnnl::memory::desc(onednn::shapeToDims(outputShape), onednn::getDataType<T>(), formatTag);

        // set up convolution operation-related descriptors
        convPrim = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, outputMemDesc,
                                            dnnl::memory::dims({stride.y, stride.x}),
                                            dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                                            dnnl::memory::dims({padBefore.y, padBefore.x}),
                                            dnnl::memory::dims({padAfter.y, padAfter.x})),
            onednn::Context::getInstance().getEngine()));
    }

   public:
    UpstrideConv2DFunctor() {}

    /**
     * @brief Sets main convolution parameters indepentent from the input, filter and output sizes
     * @param dataFormat    Expected tensors format
     * @param stride        Convolution stride
     * @param dilation      Convolution dilation
     */
    void configure(DataFormat dataFormat, const IntTuple& stride, const IntTuple& dilation) {
        this->formatTag = onednn::dataFormatToFormatTag(dataFormat);
        getSpatialStep(stride, 1, this->stride);
        getSpatialStep(dilation, 1, this->dilation);
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& filterTensor,
                    Tensor<T>& outputTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter) {
        // configure oneDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), filterTensor.getShape(), outputTensor.getShape(), padBefore, padAfter);

        // instantiate DNNL memory
        auto& engine = onednn::Context::getInstance().getEngine();
        dnnl::memory input(inputMemDesc, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory filter(filterMemDesc, engine, const_cast<T*>(filterTensor.getDataPtr()));
        dnnl::memory output(outputMemDesc, engine, outputTensor.getDataPtr());

        // g-g-go
        onednn::Context::getInstance().execute(convPrim, {{DNNL_ARG_SRC, input},
                                                          {DNNL_ARG_WEIGHTS, filter},
                                                          {DNNL_ARG_DST, output}});
    }
};

/**
 * @brief Regular 2D backward convolution implementation using oneDNN
 * 
 * @tparam T    scalar datatype 
 */
template <typename T>
class UpstrideConv2DGradFunctor<upstride::device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, kernelMemDesc, gradMemDesc, kernelGradMemDesc, inputGradMemDesc;
    dnnl::convolution_backward_data convBackDataPrim;
    dnnl::convolution_backward_weights convBackWeightsPrim;
    dnnl::memory::format_tag formatTag;
    Shape inputShape, kernelShape, gradShape;
    IntPair padBefore;  //!< zero padding: number of zeros to add at the beginning to every input spatial dimension
    IntPair padAfter;   //!< zero padding: number of zeros to add at the end to every input spatial dimension
    IntPair stride, dilation;
    bool requireInputGrad;  //!< Use to determine if inputGrad need to be compute or not

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param kernelShape       kernel tensor shape
     * @param gradShape         grad tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     */
    void configureBackend(const Shape& inputShape,
                          const Shape& kernelShape,
                          const Shape& gradShape,
                          const IntPair& padBefore,
                          const IntPair& padAfter) {
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
        kernelMemDesc = dnnl::memory::desc(onednn::shapeToDims(kernelShape), onednn::getDataType<T>(), KERNEL_MEMORY_LAYOUT);
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
                    dnnl::algorithm::convolution_direct,
                    gradMemDesc,    //  conv_bwd_src_md
                    kernelMemDesc,  // conv_diff_weights_md
                    inputMemDesc,   // conv_diff_dst_md
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
                        gradMemDesc,    //  conv_bwd_src_md
                        kernelMemDesc,  // conv_diff_weights_md
                        inputMemDesc,   // conv_diff_dst_md
                        dnnl::memory::dims({stride.y, stride.x}),
                        dnnl::memory::dims({dilation.y - 1, dilation.x - 1}),
                        dnnl::memory::dims({padBefore.y, padBefore.x}),
                        dnnl::memory::dims({padAfter.y, padAfter.x})),
                    onednn::Context::getInstance().getEngine(),
                    convPd));
        }
    }

   public:
    UpstrideConv2DGradFunctor() {}

    void configure(DataFormat dataFormat, const IntTuple& stride, const IntTuple& dilation, bool requireInputGrad) {
        this->formatTag = onednn::dataFormatToFormatTag(dataFormat);
        getSpatialStep(stride, 1, this->stride);
        getSpatialStep(dilation, 1, this->dilation);
        this->requireInputGrad = requireInputGrad;
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param kernelTensor      kernel tensor
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& kernelTensor,
                    const Tensor<const T>& gradTensor,
                    Tensor<T>& kernelGradTensor,
                    Tensor<T>& inputGradTensor,
                    const IntPair& padBefore,
                    const IntPair& padAfter) {
        // configure oneDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), kernelTensor.getShape(), gradTensor.getShape(), padBefore, padAfter);

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