#pragma once
#include "context.hpp"

namespace upstride {

template <typename T>
class UpstrideConv2DFunctor<upstride::device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, filterMemDesc, outputMemDesc;
    dnnl::convolution_forward convPrim;
    dnnl::memory::format_tag formatTag;
    Shape inputShape, filterShape, outputShape;
    IntTuple stride, padBefore, padAfter;

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     */
    void configureBackend(const Shape& inputShape, const Shape& filterShape, const Shape& outputShape, const IntTuple& padBefore, const IntTuple& padAfter) {
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
        filterMemDesc = dnnl::memory::desc(onednn::shapeToDims(filterShape), onednn::getDataType<T>(), dnnl::memory::format_tag::oihw);
        outputMemDesc = dnnl::memory::desc(onednn::shapeToDims(outputShape), onednn::getDataType<T>(), formatTag);

        // set up convolution operation-related descriptors
        // fixme: pass actual convolution parameters
        convPrim = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, outputMemDesc,
                                            dnnl::memory::dims(stride.begin(), stride.end()),
                                            dnnl::memory::dims(padBefore.begin(), padBefore.end()),
                                            dnnl::memory::dims(padAfter.begin(), padAfter.end())),
            onednn::Context::getInstance().getEngine()));
    }

   public:
    UpstrideConv2DFunctor() {}

    void configure(DataFormat dataFormat, const IntTuple& stride) {
        this->formatTag = onednn::dataFormatToFormatTag(dataFormat);
        this->stride = stride;
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
                    const IntTuple& padBefore,
                    const IntTuple& padAfter) {
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

}  // namespace upstride