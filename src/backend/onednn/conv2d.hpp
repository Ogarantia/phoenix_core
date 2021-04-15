#pragma once

#include "../backend.hpp"
#include "context.hpp"

namespace upstride {

namespace onednn {
    template<typename T>
    inline void setTensorMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape, const DataFormat format, bool forceMemTag = true) {
        desc = dnnl::memory::desc(
            { shape[0], shape.depth(format), shape.height(format), shape.width(format) },
            onednn::getDataType<T>(),
            forceMemTag ? onednn::dataFormatToFormatTag(format) : dnnl::memory::format_tag::any);
    }


    template<typename T>
    inline void setFilterMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape, const FilterLayout layout, int groups, bool forceMemTag = true) {
        const Conv2DFilterLayout filter(layout);
        if (groups == 1) {
            desc = dnnl::memory::desc(
                { filter.numOutputChannels(shape), filter.numInputChannels(shape), filter.height(shape), filter.width(shape) },
                    // dimensions are ordered in the same way regardless the filter layout
                onednn::getDataType<T>(),
                forceMemTag ? onednn::filterLayoutToFormatTag(layout, false) : dnnl::memory::format_tag::any);
        } else {
            // check validity of the number of groups
            if (filter.numOutputChannels(shape) % groups != 0)
                throw std::invalid_argument("Number of output channels is not evenly divisible in groups");
            desc = dnnl::memory::desc(
                { groups, filter.numOutputChannels(shape) / groups, filter.numInputChannels(shape), filter.height(shape), filter.width(shape) },
                onednn::getDataType<T>(),
                forceMemTag ? onednn::filterLayoutToFormatTag(layout, true) : dnnl::memory::format_tag::any);
        }
    }

    template<typename T>
    inline void setBiasMemoryDescriptor(dnnl::memory::desc& desc, const Shape& shape) {
        desc = dnnl::memory::desc(
            dnnl::memory::dims{ shape.numel() },
            onednn::getDataType<T>(),
            dnnl::memory::format_tag::x);
    }

    inline void makeReorder(dnnl::engine& engine, dnnl::reorder& primitive, dnnl::memory::desc& input, dnnl::memory::desc& output) {
        primitive = dnnl::reorder(dnnl::reorder::primitive_desc(engine, input, engine, output));
    }
}

/**
 * @brief 2D convolution implementation using oneDNN
 * @tparam T    scalar datatype
 */
template <typename T>
class ScalarConv2DFunctor<device::CPU, T> {
   private:
    /**
     * Reordering control
     * If `true`, allow oneDNN to reorder a tensor for (potentially) better speed.
     * Set empirically.
     */
    static const bool
        REORDER_INPUT = false,
        REORDER_FILTER = true,
        REORDER_OUTPUT = false;
    template<typename item_type>
    struct Bundle {
        item_type input, filter, output;
    };

    device::CPU& device;
    Bundle<dnnl::memory::desc> original;                    //!< memory descriptors of tensors given on input and output as is
    Bundle<dnnl::memory::desc> reordered;                   //!< memory descriptors of tensors after a reordering operation
    dnnl::memory::desc biasMemDesc;                         //!< bias memory descriptor
    dnnl::convolution_forward convPrimBiased, convPrim;     //!< oneDNN convolution primitives
    Bundle<dnnl::reorder> reorder;                          //!< oneDNN reorder primitives
    Bundle<Pointer> buffer;                                 //!< intermediate buffers storing reordered tensors
    Bundle<bool> needReorder;                               //!< boolean flags specifying if a reorder needed for a tensor
    const bool useBias;                                     //!< if `true`, a bias tensor is added to the convolution output

    /**
     * @brief Performs backend-related operation configuration
     * Warning: this function needs to be called in an isolated thread since it performs oneDNN resource management.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doConfigure(DataFormat dataFormat,
                     FilterLayout filterLayout,
                     const IntPair& stride,
                     const IntPair& dilation,
                     const Shape& inputShape,
                     const Shape& filterShape,
                     const Shape& biasShape,
                     const Shape& outputShape,
                     const IntPair& padBefore,
                     const IntPair& padAfter,
                     const int groups)
    {
        auto& engine = static_cast<onednn::Context&>(device.getContext()).getEngine();

        // set up original oneDNN memory descriptors
        onednn::setTensorMemoryDescriptor<T>(original.input, inputShape, dataFormat);
        onednn::setFilterMemoryDescriptor<T>(original.filter, filterShape, filterLayout, groups);
        onednn::setTensorMemoryDescriptor<T>(original.output, outputShape, dataFormat);
        if (useBias)
            onednn::setBiasMemoryDescriptor<T>(biasMemDesc, biasShape);

        // set up reordered oneDNN memory descriptors
        onednn::setTensorMemoryDescriptor<T>(reordered.input, inputShape, dataFormat, !REORDER_INPUT);
        onednn::setFilterMemoryDescriptor<T>(reordered.filter, filterShape, filterLayout, groups, !REORDER_FILTER);
        onednn::setTensorMemoryDescriptor<T>(reordered.output, outputShape, dataFormat, !REORDER_OUTPUT);

        // set up convolution operation-related descriptors
        if (useBias) {
            // biased convolution
            convPrimBiased = dnnl::convolution_forward(dnnl::convolution_forward::primitive_desc(
                dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                                dnnl::algorithm::convolution_auto,
                                                reordered.input, reordered.filter, biasMemDesc, reordered.output,
                                                dnnl::memory::dims({stride.x, stride.y}),
                                                dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                                dnnl::memory::dims({padBefore.x, padBefore.y}),
                                                dnnl::memory::dims({padAfter.x, padAfter.y})),
                engine));
        }

        // biasless convolution (it is setup anyway to be able to use both biased and biasless versions)
        auto primDesc = dnnl::convolution_forward::primitive_desc(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            reordered.input, reordered.filter, reordered.output,
                                            dnnl::memory::dims({stride.x, stride.y}),
                                            dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                            dnnl::memory::dims({padBefore.x, padBefore.y}),
                                            dnnl::memory::dims({padAfter.x, padAfter.y})),
            engine);
        convPrim = dnnl::convolution_forward(primDesc);

        // set reordered descriptors
        reordered.input = primDesc.src_desc();
        reordered.filter = primDesc.weights_desc();
        reordered.output = primDesc.dst_desc();

        // set up reordering if needed
        needReorder.input = reordered.input != original.input;
        needReorder.filter = reordered.filter != original.filter;
        needReorder.output = reordered.output != original.output;

        if (needReorder.input)
            onednn::makeReorder(engine, reorder.input, original.input, reordered.input);
        if (needReorder.filter)
            onednn::makeReorder(engine, reorder.filter, original.filter, reordered.filter);
        if (needReorder.output)
            onednn::makeReorder(engine, reorder.output, reordered.output, original.output);
    }

    /**
     * @brief Executes the convolution operation
     * Warning: this function needs to be called in an isolated thread since it uses oneDNN resources.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doCompute(const Tensor<device::CPU, const T>& inputTensor,
                   const Tensor<device::CPU, const T>& filterTensor,
                   const Tensor<device::CPU, const T>* biasTensor,
                   Tensor<device::CPU, T>& outputTensor) {
        auto& context = static_cast<onednn::Context&>(device.getContext());
        auto& engine = context.getEngine();

        // instantiate DNNL memory
        Bundle<dnnl::memory> originalMem {
            dnnl::memory(original.input, engine, const_cast<T*>(inputTensor.getDataPtr())),
            dnnl::memory(original.filter, engine, const_cast<T*>(filterTensor.getDataPtr())),
            dnnl::memory(original.output, engine, outputTensor.getDataPtr())
        };

        Bundle<dnnl::memory> reorderedMem {
            needReorder.input ? dnnl::memory(reordered.input, engine, buffer.input.data()) : originalMem.input,
            needReorder.filter ? dnnl::memory(reordered.filter, engine, buffer.filter.data()) : originalMem.filter,
            needReorder.output ? dnnl::memory(reordered.output, engine, buffer.output.data()) : originalMem.output
        };

        // execute reordering primitives if needed
        if (needReorder.input)
            context.execute(reorder.input, {{DNNL_ARG_FROM, originalMem.input},
                                            {DNNL_ARG_TO, reorderedMem.input}});
        if (needReorder.filter)
            context.execute(reorder.filter, {{DNNL_ARG_FROM, originalMem.filter},
                                             {DNNL_ARG_TO, reorderedMem.filter}});

        // execute the convolution primitive on reordered memory
        if (biasTensor) {
            if (!useBias)
                throw std::invalid_argument("Bias application is not configured for this scalar Conv2D operation");

            dnnl::memory bias(biasMemDesc, engine, const_cast<T*>(biasTensor->getDataPtr()));
            context.execute(convPrimBiased, {{DNNL_ARG_SRC, reorderedMem.input},
                                             {DNNL_ARG_WEIGHTS, reorderedMem.filter},
                                             {DNNL_ARG_BIAS, bias},
                                             {DNNL_ARG_DST, reorderedMem.output}});
        } else {
            context.execute(convPrim, {{DNNL_ARG_SRC, reorderedMem.input},
                                       {DNNL_ARG_WEIGHTS, reorderedMem.filter},
                                       {DNNL_ARG_DST, reorderedMem.output}});
        }

        // reorder output back if needed
        if (needReorder.output)
            context.execute(reorder.output, {{DNNL_ARG_FROM, reorderedMem.output},
                                             {DNNL_ARG_TO, originalMem.output}});
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
        useBias(!biasShape.empty())
    {
        // configure in an isolated thread
        device.call(this, &ScalarConv2DFunctor<device::CPU, T>::doConfigure,
            dataFormat,
            filterLayout,
            stride,
            dilation,
            inputShape,
            filterShape,
            biasShape,
            outputShape,
            padBefore,
            padAfter,
            groups);
    }

    inline void prepare(MemoryRequest& memory) {
        // allocate reordering buffers if needed
        if (needReorder.input)
            buffer.input = memory.alloc(reordered.input.get_size());
        if (needReorder.filter)
            buffer.filter = memory.alloc(reordered.filter.get_size());
        if (needReorder.output)
            buffer.output = memory.alloc(reordered.output.get_size());
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
    /**
     * Reordering control
     * If `true`, allow oneDNN to reorder a tensor for (potentially) better speed.
     * Set empirically.
     */
    static const bool
        REORDER_INPUT_FOR_FILTER_GRAD = false,      //!< reorder input before computing filter grad 
        REORDER_GRAD_FOR_FILTER_GRAD = false,       //!< reorder loss function gradient before computing filter grad
        REORDER_FILTER_GRAD = true,                //!< reorder the resulting filter gradient

        REORDER_FILTER_FOR_INPUT_GRAD = true,       //!< reorder filter before computing input grad
        REORDER_GRAD_FOR_INPUT_GRAD = false,         //!< reorder loss function gradient before computing input grad
        REORDER_INPUT_GRAD = false;                 //!< reorder the resulting input gradient

    template<typename item_type>
    struct Bundle {
        item_type input, filter, grad;
    };

    device::CPU& device;
    Bundle<dnnl::memory::desc> original;        //!< memory descriptors of tensors given on input and output as is
    Bundle<Pointer> buffer;                     //!< reordering buffers
    struct {
        Bundle<dnnl::memory::desc> reordered;   //!< reordered memory descriptors
        Bundle<bool> needReorder;               //!< memory descriptors of tensors after a reordering operation
        Bundle<dnnl::reorder> reorder;          //!< reorder primitives, original -> reordered
    } inputGrad, filterGrad;                    //!< collections of reordering facilities per output
    dnnl::convolution_backward_data convBackDataPrim;
    dnnl::convolution_backward_weights convBackWeightsPrim;
    const bool requireInputGrad;  //!< Used to determine if inputGrad needs to be computed or not

    /**
     * @brief Performs backend-related operation configuration.
     * Warning: this function needs to be called in an isolated thread since it performs oneDNN resource management.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doConfigure(DataFormat dataFormat,
                     FilterLayout filterLayout,
                     const IntPair& stride,
                     const IntPair& dilation,
                     const Shape& inputShape,
                     const Shape& filterShape,
                     const Shape& gradShape,
                     const IntPair& padBefore,
                     const IntPair& padAfter,
                     const int groups)
    {
        auto& context = static_cast<onednn::Context&>(device.getContext());
        auto& engine = context.getEngine();

        // Input, filter and grad are inputs, filterGrad and inputGrad are outputs.
        // Reordering of all inputs is done per output.

        // set up oneDNN memory descriptors
        onednn::setTensorMemoryDescriptor<T>(original.input, inputShape, dataFormat);
        onednn::setFilterMemoryDescriptor<T>(original.filter, filterShape, filterLayout, groups);
        onednn::setTensorMemoryDescriptor<T>(original.grad, gradShape, dataFormat);

        // Re-computation of the convolution forward primitive descriptor
        dnnl::convolution_forward::primitive_desc fwdConvPd(
            dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::convolution_auto,
                                            original.input, original.filter, original.grad,
                                            dnnl::memory::dims({stride.x, stride.y}),
                                            dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                                            dnnl::memory::dims({padBefore.x, padBefore.y}),
                                            dnnl::memory::dims({padAfter.x, padAfter.y})),
            engine);

        // instantiate backward conv primitive to compute the filter gradient
        // .input and .grad are primitive inputs, .filter is its output
        {
            // set up oneDNN memory descriptors
            onednn::setTensorMemoryDescriptor<T>(filterGrad.reordered.input,
                                                 inputShape,
                                                 dataFormat,
                                                 !REORDER_INPUT_FOR_FILTER_GRAD);

            onednn::setFilterMemoryDescriptor<T>(filterGrad.reordered.filter,
                                                 filterShape,
                                                 filterLayout,
                                                 groups,
                                                 !REORDER_FILTER_GRAD);

            onednn::setTensorMemoryDescriptor<T>(filterGrad.reordered.grad,
                                                 gradShape,
                                                 dataFormat,
                                                 !REORDER_GRAD_FOR_FILTER_GRAD);

            auto primDesc = dnnl::convolution_backward_weights::primitive_desc(
                dnnl::convolution_backward_weights::desc(
                    dnnl::algorithm::convolution_auto,
                    filterGrad.reordered.input,   // conv_diff_dst_md
                    filterGrad.reordered.filter,  // conv_diff_weights_md
                    filterGrad.reordered.grad,    // conv_bwd_src_md
                    dnnl::memory::dims({stride.x, stride.y}),
                    dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                    dnnl::memory::dims({padBefore.x, padBefore.y}),
                    dnnl::memory::dims({padAfter.x, padAfter.y})),
                engine,
                fwdConvPd);
            convBackWeightsPrim = dnnl::convolution_backward_weights(primDesc);

            // fetch reordered descriptors back with format tags defined
            filterGrad.reordered.input = primDesc.src_desc();           // primitive input
            filterGrad.reordered.filter = primDesc.diff_weights_desc(); // primitive output
            filterGrad.reordered.grad = primDesc.diff_dst_desc();       // primitive input

            // set up reordering if needed
            filterGrad.needReorder.input = filterGrad.reordered.input != original.input;
            filterGrad.needReorder.filter = filterGrad.reordered.filter != original.filter;
            filterGrad.needReorder.grad = filterGrad.reordered.grad != original.grad;

            if (filterGrad.needReorder.input)
                onednn::makeReorder(engine, filterGrad.reorder.input, original.input, filterGrad.reordered.input);
            if (filterGrad.needReorder.filter)
                onednn::makeReorder(engine, filterGrad.reorder.filter, filterGrad.reordered.filter, original.filter);
            if (filterGrad.needReorder.grad)
                onednn::makeReorder(engine, filterGrad.reorder.grad, original.grad, filterGrad.reordered.grad);
        }

        // instantiate backward conv primitive to compute the input gradient (only if required)
        // .filter and .grad are primitive inputs, .input is the resulting input gradient
        if (requireInputGrad) {
            // set up oneDNN memory descriptors
            onednn::setTensorMemoryDescriptor<T>(inputGrad.reordered.input,
                                                 inputShape,
                                                 dataFormat,
                                                 !REORDER_INPUT_GRAD);

            onednn::setFilterMemoryDescriptor<T>(inputGrad.reordered.filter,
                                                 filterShape,
                                                 filterLayout,
                                                 groups,
                                                 !REORDER_FILTER_FOR_INPUT_GRAD);

            onednn::setTensorMemoryDescriptor<T>(inputGrad.reordered.grad,
                                                 gradShape,
                                                 dataFormat,
                                                 !REORDER_GRAD_FOR_INPUT_GRAD);

            // instantiate backward conv primitive to compute the input gradient
            auto primDesc = dnnl::convolution_backward_data::primitive_desc(
                dnnl::convolution_backward_data::desc(
                    dnnl::algorithm::convolution_auto,
                    inputGrad.reordered.input,   // conv_diff_dst_md
                    inputGrad.reordered.filter,  // conv_diff_weights_md
                    inputGrad.reordered.grad,    // conv_bwd_src_md
                    dnnl::memory::dims({stride.x, stride.y}),
                    dnnl::memory::dims({dilation.x - 1, dilation.y - 1}),
                    dnnl::memory::dims({padBefore.x, padBefore.y}),
                    dnnl::memory::dims({padAfter.x, padAfter.y})),
                engine,
                fwdConvPd);
            convBackDataPrim = dnnl::convolution_backward_data(primDesc);

            // fetch reordered descriptors back with format tags defined
            inputGrad.reordered.input = primDesc.diff_src_desc();
            inputGrad.reordered.filter = primDesc.weights_desc();
            inputGrad.reordered.grad = primDesc.diff_dst_desc();

            // set up reordering if needed
            inputGrad.needReorder.input = inputGrad.reordered.input != original.input;
            inputGrad.needReorder.filter = inputGrad.reordered.filter != original.filter;
            inputGrad.needReorder.grad = inputGrad.reordered.grad != original.grad;

            if (inputGrad.needReorder.input)
                onednn::makeReorder(engine, inputGrad.reorder.input, inputGrad.reordered.input, original.input);
            if (inputGrad.needReorder.filter)
                onednn::makeReorder(engine, inputGrad.reorder.filter, original.filter, inputGrad.reordered.filter);
            if (inputGrad.needReorder.grad)
                onednn::makeReorder(engine, inputGrad.reorder.grad, original.grad, inputGrad.reordered.grad);
        }

        else {
            inputGrad.needReorder.input = false;
            inputGrad.needReorder.filter = false;
            inputGrad.needReorder.grad = false;
        }

    }

    /**
     * @brief Executes the convolution operation.
     * Warning: this function needs to be called in an isolated thread since it uses oneDNN resources.
     * Calling it form a user thread may end up with a segmentation fault.
     */
    void doCompute(const Tensor<device::CPU, const T>& inputTensor,
                   const Tensor<device::CPU, const T>& filterTensor,
                   const Tensor<device::CPU, const T>& gradTensor,
                   Tensor<device::CPU, T>& filterGradTensor,
                   Tensor<device::CPU, T>& inputGradTensor) {
        // instantiate DNNL memory
        auto& context = static_cast<onednn::Context&>(device.getContext());
        auto& engine = context.getEngine();

        dnnl::memory originalInput(original.input, engine, const_cast<T*>(inputTensor.getDataPtr()));
        dnnl::memory originalFilter(original.filter, engine, const_cast<T*>(filterTensor.getDataPtr()));
        dnnl::memory originalGrad(original.grad, engine, const_cast<T*>(gradTensor.getDataPtr()));
        dnnl::memory originalInputGrad(original.input, engine, inputGradTensor.getDataPtr());
        dnnl::memory originalFilterGrad(original.filter, engine, filterGradTensor.getDataPtr());

        // filter gradient computation
        {
            // prepare descriptors for filter gradient computation
            dnnl::memory grad   = filterGrad.needReorder.grad ?
                                  dnnl::memory(filterGrad.reordered.grad,   engine, buffer.grad.data())   : originalGrad;
            dnnl::memory input  = filterGrad.needReorder.input ?
                                  dnnl::memory(filterGrad.reordered.input,  engine, buffer.input.data())  : originalInput;
            dnnl::memory filter = filterGrad.needReorder.filter ?
                                  dnnl::memory(filterGrad.reordered.filter, engine, buffer.filter.data()) : originalFilterGrad;

            // reorder, if needed
            if (filterGrad.needReorder.grad)
                context.execute(filterGrad.reorder.grad, {{DNNL_ARG_FROM, originalGrad},
                                                          {DNNL_ARG_TO, grad}});
            if (filterGrad.needReorder.input)
                context.execute(filterGrad.reorder.input, {{DNNL_ARG_FROM, originalInput},
                                                           {DNNL_ARG_TO, input}});
            // g-g-go
            context.execute(convBackWeightsPrim, {{DNNL_ARG_SRC, input},
                                                  {DNNL_ARG_DIFF_DST, grad},
                                                  {DNNL_ARG_DIFF_WEIGHTS, filter}});

            // reorder back if needed
            if (filterGrad.needReorder.filter)
                context.execute(filterGrad.reorder.filter, {{DNNL_ARG_FROM, filter},
                                                            {DNNL_ARG_TO, originalFilterGrad}});
        }

        // input gradient computation
        if (requireInputGrad) {
            // prepare descriptors for input gradient computation
            dnnl::memory grad   = inputGrad.needReorder.grad ?
                                  dnnl::memory(inputGrad.reordered.grad,   engine, buffer.grad.data())   : originalGrad;
            dnnl::memory input  = inputGrad.needReorder.input ?
                                  dnnl::memory(inputGrad.reordered.input,  engine, buffer.input.data())  : originalInputGrad;
            dnnl::memory filter = inputGrad.needReorder.filter ?
                                  dnnl::memory(inputGrad.reordered.filter, engine, buffer.filter.data()) : originalFilter;

            // reorder, if needed
            if (inputGrad.needReorder.grad)
                context.execute(inputGrad.reorder.grad, {{DNNL_ARG_FROM, originalGrad},
                                                         {DNNL_ARG_TO, grad}});
            if (inputGrad.needReorder.filter)
                context.execute(inputGrad.reorder.filter, {{DNNL_ARG_FROM, originalFilter},
                                                           {DNNL_ARG_TO, filter}});

            // g-g-go
            context.execute(convBackDataPrim, {{DNNL_ARG_WEIGHTS, filter},
                                               {DNNL_ARG_DIFF_DST, grad},
                                               {DNNL_ARG_DIFF_SRC, input}});

            // reorder back if needed
            if (inputGrad.needReorder.input)
                context.execute(inputGrad.reorder.input, {{DNNL_ARG_FROM, input},
                                                          {DNNL_ARG_TO, originalInputGrad}});
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
        requireInputGrad(requireInputGrad)
    {
        device.call(this, &ScalarConv2DGradFunctor<device::CPU, T>::doConfigure,
            dataFormat,
            filterLayout,
            stride,
            dilation,
            inputShape,
            filterShape,
            outputShape,
            padBefore,
            padAfter,
            groups);
    }

    inline void prepare(MemoryRequest& memory) {
        // allocate reordering buffers if needed
        if (filterGrad.needReorder.input || inputGrad.needReorder.input) {
            const size_t
                a = filterGrad.needReorder.input ? filterGrad.reordered.input.get_size() : 0,
                b = inputGrad.needReorder.input ? inputGrad.reordered.input.get_size() : 0;
            buffer.input = memory.alloc(std::max(a, b));
        }
        if (filterGrad.needReorder.filter || inputGrad.needReorder.filter) {
            const size_t
                a = filterGrad.needReorder.filter ? filterGrad.reordered.filter.get_size() : 0,
                b = inputGrad.needReorder.filter ? inputGrad.reordered.filter.get_size() : 0;
            buffer.filter = memory.alloc(std::max(a, b));
        }
        if (filterGrad.needReorder.grad || inputGrad.needReorder.grad) {
            const size_t
                a = filterGrad.needReorder.grad ? filterGrad.reordered.grad.get_size() : 0,
                b = inputGrad.needReorder.grad ? inputGrad.reordered.grad.get_size() : 0;
            buffer.grad = memory.alloc(std::max(a, b));
        }
    }

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