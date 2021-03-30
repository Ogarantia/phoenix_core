#pragma once

#include "../backend.hpp"
#include "context.hpp"

namespace upstride {

namespace onednn {
    template<typename T>
    inline void setTensorMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape, const DataFormat format) {
        desc = dnnl::memory::desc(
            { shape[0], shape.depth(format), shape.height(format), shape.width(format) },
            onednn::getDataType<T>(),
            onednn::dataFormatToFormatTag(format));
    }


    template<typename T>
    inline void setFilterMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape, const FilterLayout layout, int groups) {
        const Conv2DFilterLayout filter(layout);
        if (groups == 1) {
            desc = dnnl::memory::desc(
                { filter.numOutputChannels(shape), filter.numInputChannels(shape), filter.height(shape), filter.width(shape) },
                    // dimensions are ordered in the same way regardless the filter layout
                onednn::getDataType<T>(),
                onednn::filterLayoutToFormatTag(layout));
        } else {
            // check validity of the number of groups
            if (filter.numOutputChannels(shape) % groups != 0)
                throw std::invalid_argument("Number of output channels is not evenly divisible in groups");

            // get filter memory tag
            dnnl::memory::format_tag memoryTag;
            switch (layout) {
            case FilterLayout::OHWI:
                memoryTag = dnnl::memory::format_tag::gohwi;
                break;
            case FilterLayout::OIHW:
                memoryTag = dnnl::memory::format_tag::goihw;
                break;
            case FilterLayout::HWIO:
                memoryTag = dnnl::memory::format_tag::hwigo;
                break;
            default:
                throw std::invalid_argument("Cannot infer group conv2d filter layout");
            }

            desc = dnnl::memory::desc(
                { groups, filter.numOutputChannels(shape) / groups, filter.numInputChannels(shape), filter.height(shape), filter.width(shape) },
                onednn::getDataType<T>(),
                memoryTag);
        }
    }

    template<typename T>
    inline void setBiasMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape) {
        desc = dnnl::memory::desc(
            dnnl::memory::dims{ shape.numel() },
            onednn::getDataType<T>(),
            dnnl::memory::format_tag::x);
    }
}

/**
 * @brief 2D convolution implementation using oneDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class ScalarConv2DFunctor<device::CPU, T> {
   private:
    device::CPU& device;
    dnnl::memory::desc inputMemDesc, filterMemDesc, biasMemDesc, outputMemDesc;
    dnnl::convolution_forward convPrim, convPrimNoBias;
    const IntPair stride, dilation;
    const bool useBias;  //!< if `true`, a bias tensor is added to the convolution output

    /**
     * @brief Performs backend-related operation configuration
     * Warning: this function needs to be called in an isolated thread since it performs oneDNN resource management.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doConfigure(DataFormat dataFormat,
                     FilterLayout filterLayout,
                     const Shape& inputShape,
                     const Shape& filterShape,
                     const Shape& biasShape,
                     const Shape& outputShape,
                     const IntPair& padBefore,
                     const IntPair& padAfter,
                     const int groups) {
        auto& engine = static_cast<onednn::Context&>(device.getContext()).getEngine();

        // set up oneDNN memory descriptors
        onednn::setTensorMemoryDescriptor<T>(inputMemDesc, inputShape, dataFormat);
        onednn::setFilterMemoryDescriptor<T>(filterMemDesc, filterShape, filterLayout, groups);
        if (useBias)
            onednn::setBiasMemoryDescriptor<T>(biasMemDesc, biasShape);
        onednn::setTensorMemoryDescriptor<T>(outputMemDesc, outputShape, dataFormat);

        // set up convolution operation-related descriptors
        if (useBias) {
            // biased convolution
            convPrim = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
                dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                                dnnl::algorithm::convolution_auto,
                                                inputMemDesc, filterMemDesc, biasMemDesc, outputMemDesc,
                                                dnnl::memory::dims({stride.x, stride.y}),
                                                dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                                dnnl::memory::dims({padBefore.x, padBefore.y}),
                                                dnnl::memory::dims({padAfter.x, padAfter.y})),
                engine));
        }

        // biasless convolution (it is setup anyway to be able to use both biased and biasless versions)
        convPrimNoBias = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, outputMemDesc,
                                            dnnl::memory::dims({stride.x, stride.y}),
                                            dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                            dnnl::memory::dims({padBefore.x, padBefore.y}),
                                            dnnl::memory::dims({padAfter.x, padAfter.y})),
            engine));
    }

    /**
     * @brief Executes the convolution operation
     * Warning: this function needs to be called in an isolated thread since it uses oneDNN resources.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doCompute(const Tensor<device::CPU, const T>& inputTensor,
                   const Tensor<device::CPU, const T>& kernelTensor,
                   const Tensor<device::CPU, const T>* biasTensor,
                   Tensor<device::CPU, T>& outputTensor) {
        // instantiate DNNL memory
        auto& context = static_cast<onednn::Context&>(device.getContext());
        auto& engine = context.getEngine();
        dnnl::memory input(inputMemDesc, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory kernel(filterMemDesc, engine, const_cast<T*>(kernelTensor.getDataPtr()));
        dnnl::memory output(outputMemDesc, engine, outputTensor.getDataPtr());

        if (biasTensor) {
            if (!useBias)
                throw std::invalid_argument("Bias application is not configured for this scalar Conv2D operation");

            dnnl::memory bias(biasMemDesc, engine, const_cast<T*>(biasTensor->getDataPtr()));
            context.execute(convPrim, {{DNNL_ARG_SRC, input},
                                       {DNNL_ARG_WEIGHTS, kernel},
                                       {DNNL_ARG_BIAS, bias},
                                       {DNNL_ARG_DST, output}});
        } else {
            context.execute(convPrimNoBias, {{DNNL_ARG_SRC, input},
                                             {DNNL_ARG_WEIGHTS, kernel},
                                             {DNNL_ARG_DST, output}});
        }
    }

   public:
    /**
     * @brief Instantiates a Conv2D operation.
     * @param device            A device the operation will be executed on
     * @param dataFormat        Memory layout of input and output tensors
     * @param filterLayout      Convolution filter layout
     * @param stride            Convolution stride
     * @param dilation          Convolution dilation
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param biasShape         Bias tensor shape (empty if no bias addition is enabled)
     * @param outputShape       Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     */
    ScalarConv2DFunctor(
        device::CPU& device,
        DataFormat dataFormat,
        FilterLayout filterLayout,
        const IntPair& stride,
        const IntPair& dilation,
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& biasShape,
        const Shape& outputShape,
        const IntPair& padBefore,
        const IntPair& padAfter,
        int groups
    ) :
        device(device),
        stride(stride),
        dilation(dilation),
        useBias(!biasShape.empty())
    {
        // configure in an isolated thread
        device.call(this, &ScalarConv2DFunctor<device::CPU, T>::doConfigure,
            dataFormat,
            filterLayout,
            inputShape,
            filterShape,
            biasShape,
            outputShape,
            padBefore,
            padAfter,
            groups);
    }

    inline void prepare(MemoryRequest& memory) {}

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
        device.call(this, &ScalarConv2DFunctor<device::CPU, T>::doCompute, inputTensor, kernelTensor, biasTensor, outputTensor);
    }
};

/**
 * @brief 2D backward convolution implementation using oneDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class ScalarConv2DGradFunctor<device::CPU, T> {
   private:
    device::CPU& device;
    dnnl::memory::desc inputMemDesc, filterMemDesc, gradMemDesc, filterGradMemDesc, inputGradMemDesc;
    dnnl::convolution_backward_data convBackDataPrim;
    dnnl::convolution_backward_weights convBackWeightsPrim;
    const IntPair stride, dilation;
    const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not

    /**
     * @brief Performs backend-related operation configuration.
     * Warning: this function needs to be called in an isolated thread since it performs oneDNN resource management.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doConfigure(DataFormat dataFormat,
                     FilterLayout filterLayout,
                     const Shape& inputShape,
                     const Shape& filterShape,
                     const Shape& gradShape,
                     const IntPair& padBefore,
                     const IntPair& padAfter,
                     const int groups)
    {
        auto& context = static_cast<onednn::Context&>(device.getContext());

        // set up oneDNN memory descriptors
        onednn::setTensorMemoryDescriptor<T>(inputMemDesc, inputShape, dataFormat);
        onednn::setFilterMemoryDescriptor<T>(filterMemDesc, filterShape, filterLayout, groups);
        onednn::setFilterMemoryDescriptor<T>(filterGradMemDesc, filterShape, filterLayout, groups);
        onednn::setTensorMemoryDescriptor<T>(gradMemDesc, gradShape, dataFormat);
        onednn::setTensorMemoryDescriptor<T>(inputGradMemDesc, inputShape, dataFormat);

        // Re-computation of the convolution forward primitive descriptor
        dnnl::convolution_forward::primitive_desc convPd(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            inputMemDesc, filterMemDesc, gradMemDesc,
                                            dnnl::memory::dims({stride.x, stride.y}),
                                            dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                            dnnl::memory::dims({padBefore.x, padBefore.y}),
                                            dnnl::memory::dims({padAfter.x, padAfter.y})),
            context.getEngine());

        // instantiate backward conv primitive to compute the kernel gradient
        convBackWeightsPrim = dnnl::convolution_backward_weights(
            dnnl::convolution_backward_weights::primitive_desc(
                dnnl::convolution_backward_weights::desc(
                    dnnl::algorithm::convolution_auto,
                    inputMemDesc,   // conv_diff_dst_md
                    filterMemDesc,  // conv_diff_weights_md
                    gradMemDesc,    //  conv_bwd_src_md
                    dnnl::memory::dims({stride.x, stride.y}),
                    dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                    dnnl::memory::dims({padBefore.x, padBefore.y}),
                    dnnl::memory::dims({padAfter.x, padAfter.y})),
                context.getEngine(),
                convPd));

        // instantiate backward conv primitive to compute the input gradient (only if required)
        if (requireInputGrad) {
            convBackDataPrim = dnnl::convolution_backward_data(
                dnnl::convolution_backward_data::primitive_desc(
                    dnnl::convolution_backward_data::desc(
                        dnnl::algorithm::convolution_direct,
                        inputMemDesc,   // conv_diff_dst_md
                        filterMemDesc,  // conv_diff_weights_md
                        gradMemDesc,    //  conv_bwd_src_md
                        dnnl::memory::dims({stride.x, stride.y}),
                        dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                        dnnl::memory::dims({padBefore.x, padBefore.y}),
                        dnnl::memory::dims({padAfter.x, padAfter.y})),
                    context.getEngine(),
                    convPd));
        }
    }

    /**
     * @brief Executes the convolution operation.
     * Warning: this function needs to be called in an isolated thread since it uses oneDNN resources.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doCompute(const Tensor<device::CPU, const T>& inputTensor,
                   const Tensor<device::CPU, const T>& kernelTensor,
                   const Tensor<device::CPU, const T>& gradTensor,
                   Tensor<device::CPU, T>& kernelGradTensor,
                   Tensor<device::CPU, T>& inputGradTensor) {
        // instantiate DNNL memory
        auto& context = static_cast<onednn::Context&>(device.getContext());
        auto& engine = context.getEngine();

        dnnl::memory input(inputMemDesc, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory kernel(filterMemDesc, engine, const_cast<T*>(kernelTensor.getDataPtr()));
        dnnl::memory grad(gradMemDesc, engine, const_cast<T*>(gradTensor.getDataPtr()));

        dnnl::memory kernelGrad(filterGradMemDesc, engine, kernelGradTensor.getDataPtr());
        dnnl::memory inputGrad(inputGradMemDesc, engine, inputGradTensor.getDataPtr());

        // g-g-go
        context.execute(convBackWeightsPrim, {{DNNL_ARG_SRC, input},
                                              {DNNL_ARG_DIFF_DST, grad},
                                              {DNNL_ARG_DIFF_WEIGHTS, kernelGrad}}  // output
        );
        if (requireInputGrad) {
            context.execute(convBackDataPrim, {{DNNL_ARG_WEIGHTS, kernel},
                                               {DNNL_ARG_DIFF_DST, grad},
                                               {DNNL_ARG_DIFF_SRC, inputGrad}}  //output
            );
        }
    }

   public:
    /**
     * @brief Instantiates a Conv2D backward operation.
     * @param device            A device the operation will be executed on
     * @param dataFormat        Memory layout of input and output tensors
     * @param filterLayout      Convolution filter layout
     * @param stride            Convolution stride
     * @param dilation          Convolution dilation
     * @param inputShape        Input tensor shape
     * @param filterShape       Filter tensor shape
     * @param outputShape       Output tensor shape
     * @param padBefore         Number of zero samples to add to the input tensor on top/left
     * @param padAfter          Number of zero samples to add to the input tensor on bottom/right
     * @param groups            Number of groups for depthwise / grouped convolutions
     * @param requireInputGrad  If `true`, the gradient with respect to the input tensor is computed
     */
    ScalarConv2DGradFunctor(
        device::CPU& device,
        DataFormat dataFormat,
        FilterLayout filterLayout,
        const IntPair& stride,
        const IntPair& dilation,
        const Shape& inputShape,
        const Shape& filterShape,
        const Shape& outputShape,
        const IntPair& padBefore,
        const IntPair& padAfter,
        int groups,
        bool requireInputGrad
    ) :
        device(device),
        stride(stride),
        dilation(dilation),
        requireInputGrad(requireInputGrad)
    {
        device.call(this, &ScalarConv2DGradFunctor<device::CPU, T>::doConfigure,
            dataFormat,
            filterLayout,
            inputShape,
            filterShape,
            outputShape,
            padBefore,
            padAfter,
            groups);
    }

    inline void prepare(MemoryRequest& memory) {}

    /**
     * @brief Executes the convolution operation
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
        device.call(this, &ScalarConv2DGradFunctor<device::CPU, T>::doCompute, inputTensor, kernelTensor, gradTensor, kernelGradTensor, inputGradTensor);
    }
};

}  // namespace upstride