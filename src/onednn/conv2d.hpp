#pragma once
#include "context.hpp"

namespace upstride {

template <typename T>
class UpstrideConv2DFunctor<upstride::device::CPU, T> {
   private:
    dnnl::memory::desc inputMemDesc, filterMemDesc, outputMemDesc;
    dnnl::convolution_forward convPrim;
    const dnnl::memory::data_type dataType;
    dnnl::memory::format_tag formatTag;
    Shape inputShape, filterShape, outputShape;

    /**
     * @brief Performs backend-related operation configuration
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     */
    void configureBackend(
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& outputTensor) {
        // check if up-to-date
        if (this->inputShape == inputShape && this->filterShape == filterShape && this->outputShape == outputShape)
            return;

        // cache shapes for further up-to-dateness checks
        this->inputShape = inputShape;
        this->filterShape = filterShape;
        this->outputShape = outputShape;

        // set up oneDNN memory descriptors
        inputMemDesc = dnnl::memory::desc(dnnl::memory::dims(inputShape.getShapePtr(), inputShape.getShapePtr() + inputShape.getSize()),
                                          dataType, formatTag);
        filterMemDesc = dnnl::memory::desc(dnnl::memory::dims(filterShape.getShapePtr(), filterShape.getShapePtr() + filterShape.getSize()),
                                           dataType, dnnl::memory::format_tag::oihw);
        outputMemDesc = dnnl::memory::desc(dnnl::memory::dims(outputTensor.getShapePtr(), outputTensor.getShapePtr() + outputTensor.getSize()),
                                           dataType, formatTag);

        // set up convolution operation-related descriptors
        // fixme: pass actual convolution parameters
        convPrim = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, outputMemDesc,
                                            dnnl::memory::dims{1, 1},
                                            dnnl::memory::dims{0, 0}, dnnl::memory::dims{0, 0}),
            onednn::Context::getInstance().getEngine()));
    }

   public:
    UpstrideConv2DFunctor() : dataType(onednn::getDataType<T>()) {}

    void configure(DataFormat dataFormat) {
        formatTag = onednn::convertDataFormatToFormatTag(dataFormat);
    }

    /**
     * @brief Executes the convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<const T>& inputTensor,
                    const Tensor<const T>& filterTensor,
                    Tensor<T>& outputTensor) {
        // configure oneDNN-related stuff in a deferred fashion
        configureBackend(inputTensor.getShape(), filterTensor.getShape(), outputTensor.getShape());

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