
#include <chrono>
#include <iostream>

#include "dnnl.hpp"
#include "onednn/context.hpp"
#include "upstride.hpp"

static upstride::onednn::Context context(1);  //fixme

/**
 * @brief oneDNN convolution operation state
 */
class upstride::UpstrideConv2DFunctor<upstride::device::CPU, float>::Backend {
   private:
    dnnl::memory::desc inputMemDesc, filterMemDesc, outputMemDesc;
    dnnl::convolution_forward::desc convDesc;
    dnnl::convolution_forward::primitive_desc convPrimDesc;
    dnnl::convolution_forward convPrim;

   public:
    /**
     * @brief Configures convolution operation
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputTensor      Output tensor shape
     * @param tensorFormat      Input and output data formats
     */
    Backend(
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& outputTensor,
        DataFormat tensorFormat) : inputMemDesc(dnnl::memory::dims(inputShape.getShapePtr(), inputShape.getShapePtr() + inputShape.getSize()),
                                                onednn::getDataType<float>(),
                                                onednn::convertDataFormatToFormatTag(tensorFormat)),
                                   filterMemDesc(dnnl::memory::dims(filterShape.getShapePtr(), filterShape.getShapePtr() + filterShape.getSize()),
                                                 onednn::getDataType<float>(),
                                                 dnnl::memory::format_tag::oihw),
                                   outputMemDesc(dnnl::memory::dims(outputTensor.getShapePtr(), outputTensor.getShapePtr() + outputTensor.getSize()),
                                                 onednn::getDataType<float>(),
                                                 onednn::convertDataFormatToFormatTag(tensorFormat)),

                                   convDesc(dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, outputMemDesc,
                                            dnnl::memory::dims{1, 1},
                                            dnnl::memory::dims{0, 0}, dnnl::memory::dims{0, 0}),
                                   // fixme: pass actual convolution parameters
                                   convPrimDesc(convDesc, context.getEngine()),
                                   convPrim(convPrimDesc) {}

    /**
     * @brief Executes convolution operation
     * @param inputTensor       Input tensor
     * @param filterTensor      Filter tensor
     * @param outputTensor      Output tensor
     */
    void operator()(const Tensor<const float>& inputTensor,
                    const Tensor<const float>& filterTensor,
                    Tensor<float>& outputTensor) {
        dnnl::memory input(inputMemDesc, context.getEngine(), const_cast<float*>(inputTensor.getDataPtr()));
        dnnl::memory filter(filterMemDesc, context.getEngine(), const_cast<float*>(filterTensor.getDataPtr()));
        dnnl::memory output(outputMemDesc, context.getEngine(), outputTensor.getDataPtr());
        context.execute(convPrim, {{DNNL_ARG_SRC, input},
                                   {DNNL_ARG_WEIGHTS, filter},
                                   {DNNL_ARG_DST, output}});
    }
};


void upstride::UpstrideConv2DFunctor<upstride::device::CPU, float>::operator()(
    const Tensor<const float>& inputTensor,
    const Tensor<const float>& filterTensor,
    Tensor<float>& outputTensor,
    DataFormat dataFormat) {
    // fixme: check if the backend is up-to-date
    if (!backend) {
        backend = new upstride::UpstrideConv2DFunctor<upstride::device::CPU, float>::Backend(
            inputTensor.getShape(), filterTensor.getShape(), outputTensor.getShape(),
            dataFormat);
    }

    (*backend)(inputTensor, filterTensor, outputTensor);
}

upstride::UpstrideConv2DFunctor<upstride::device::CPU, float>::~UpstrideConv2DFunctor() {
    delete backend;
}