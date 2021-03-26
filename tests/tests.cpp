#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdlib>  // calloc
#include <iostream>  // cout
#include <random>

#include "doctest/doctest.h"
#include "upstride.hpp"

#ifdef BACKEND_CUDNN
#include <cuda.h>
#include "backend/cudnn/kernels.hpp"
#endif


template<typename Device>
inline upstride::Context& getContextInstance();

template<>
inline upstride::Context& getContextInstance<upstride::device::CPU>() {
    static upstride::onednn::Context context;
    return context;
}

#ifdef BACKEND_CUDNN
template<>
inline upstride::Context& getContextInstance<upstride::device::CUDA>() {
    static upstride::cudnn::Context context;
    return context;
}
#endif

/**
 * @brief Fill a tensor with random floating values.
 *
 * @param t Tensor to fill
 */
void setRandVal(upstride::AllocatedTensor<upstride::device::CPU, float>& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);
    for (unsigned i = 0; i < t.getShape().numel(); i++) {
        t.getDataPtr()[i] = dist(gen);
    }
}

void setRandVal(upstride::AllocatedTensor<upstride::device::CPU, int>& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-10, 10);
    for (unsigned i = 0; i < t.getShape().numel(); i++) {
        t.getDataPtr()[i] = dist(gen);
    }
}


/**
 * @brief Performs element-wise comparison of two tensors.
 * @tparam T            Tensors scalar datatype
 * @param lhs           Left-hand side tensor
 * @param rhs           Right-hand side tensor
 * @param threshold     A threshold
 * @return true if tensors are of the same shape and their values do not differ by more than the the threshold in absolute value,
 * @return false otherwise.
 */
template<typename T, typename Device>
bool compareTensors(const upstride::Tensor<Device, T>& lhs, upstride::Tensor<Device, T>& rhs, const T threshold = (T)0) {
    if (lhs.getShape() != rhs.getShape())
        return false;
    std::vector<T> lhsContents(lhs), rhsContents(rhs);
    for (size_t i = 0; i < lhsContents.size(); ++i)
        if (std::abs(lhsContents[i] - rhsContents[i]) > threshold)
            return false;
    return true;
}


TEST_CASE("Test:Shape") {
    std::cout << " -- Shape tests..." << std::endl;

    SUBCASE("Shape::Shape()") {
        std::cout << "    Shape::Shape()" << std::endl;
        upstride::Shape shape({1, 2, 3, 4});
        auto ptr = shape.getShapePtr();
        CHECK((ptr[0] == 1 && ptr[1] == 2 && ptr[2] == 3 && ptr[3] == 4));
    }

    SUBCASE("Shape == Shape") {
        std::cout << "    Shape::operator==" << std::endl;
        const upstride::Shape s1({1, 2, 3, 4});
        const upstride::Shape s2({1, 2, 3, 4});
        CHECK((s1 == s2));
    }

    SUBCASE("Shape != Shape") {
        std::cout << "    Shape::operator!=" << std::endl;
        const upstride::Shape s1({1, 2, 3, 4});
        const upstride::Shape s2({1, 2, 6, 4});
        const upstride::Shape s3({1, 2});
        CHECK((s1 != s2));
        CHECK((s1 != s3));
    }

    SUBCASE("Shape slicing and splitting") {
        std::cout << " Shape slicing and splitting" << std::endl;
        const upstride::Shape testShape{0, 1, 2, 3, 4, 5, 6};

        CHECK((testShape.slice(0, 3) == upstride::Shape{0, 1, 2}));
        CHECK((testShape.slice(4) == upstride::Shape{4, 5, 6}));
        CHECK((testShape.slice(-3) == upstride::Shape{4, 5, 6}));
        CHECK((testShape.slice(-3, -1) == upstride::Shape{4, 5}));

        CHECK((testShape.slice(2, 3).split(1) == upstride::Shape{2}));
        CHECK((testShape.slice(3).split(3) == upstride::Shape{1, 4, 5, 6}));
        CHECK((testShape.slice(4).split(2) == upstride::Shape{2, 5, 6}));

    }
}

enum binop { plus,
             minus };
template <typename T>
bool accumulatorTest(const upstride::AllocatedTensor<upstride::device::CPU, T>& srcTensor, upstride::AllocatedTensor<upstride::device::CPU, T>& dstTensor, binop op) {
    upstride::AllocatedTensor<upstride::device::CPU, T> dstCopyTensor(dstTensor.getDevice(), dstTensor.getShape());

    for (unsigned i = 0; i < dstTensor.getShape().numel(); i++) {
        dstCopyTensor.getDataPtr()[i] = dstTensor.getDataPtr()[i];
    }
    if (op == binop::plus) {
        dstTensor += srcTensor;
        for (unsigned i = 0; i < dstTensor.getShape().numel(); i++) {
            if (dstTensor.getDataPtr()[i] != dstCopyTensor.getDataPtr()[i] + srcTensor.getDataPtr()[i]) {
                return false;
            }
        }
    } else if (op == binop::minus) {
        dstTensor -= srcTensor;
        for (unsigned i = 0; i < dstTensor.getShape().numel(); i++) {
            if (dstTensor.getDataPtr()[i] != dstCopyTensor.getDataPtr()[i] - srcTensor.getDataPtr()[i]) {
                return false;
            }
        }
    } else {
        std::cout << "Error binary operation expected is \"plus\" or \"minus\"" << std::endl;
        return false;
    }
    return true;
}

TEST_CASE("Test:Tensor") {
    static upstride::onednn::Context context;
    static upstride::device::CPU device(context);

    std::cout << " -- Tensor tests..." << std::endl;
    int H = 224, W = 224, C = 3;
    int numel = H * W * C;
    upstride::Shape s1({1, C, H, W});

    SUBCASE("Tensor::Tensor()") {
        std::cout << "    Tensor::Tensor()" << std::endl;

        upstride::AllocatedTensor<upstride::device::CPU, float> t1(device, s1);

        float* t1Ptr = t1.getDataPtr();
        // Ensure values are not zero
        for (int i = 0; i < numel; i++)
            t1Ptr[i] = i * 3.0f;

        // Verify if values were indeed modified before applying t1.zero()
        bool test = true;
        for (int i = 0; i < numel && test; i++) {
            if (t1Ptr[i] != i * 3.0f)
                test = false;
        }

        t1.zero();
        // Verify if new values are zero
        for (int i = 0; i < numel && test; i++) {
            if (t1Ptr[i] != 0.0f)
                test = false;
        }
        CHECK((test));
    }

    SUBCASE("[float/int] Tensor Src + Dst") {
        std::cout << "    [float] Test: Tensor Src + Dst" << std::endl;
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(device, s1), dstTensori(device, s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(device, s1), dstTensorf(device, s1);
        setRandVal(srcTensori);
        setRandVal(dstTensori);
        setRandVal(srcTensorf);
        setRandVal(dstTensorf);

        CHECK((accumulatorTest(srcTensori, dstTensori, binop::plus)));
        CHECK((accumulatorTest(srcTensorf, dstTensorf, binop::plus)));
    }

    SUBCASE("[float/int] Tensor Src - Dst") {
        std::cout << "    [float/int] Test: Tensor Src - Dst" << std::endl;
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(device, s1), dstTensori(device, s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(device, s1), dstTensorf(device, s1);
        setRandVal(srcTensori);
        setRandVal(dstTensori);
        setRandVal(srcTensorf);
        setRandVal(dstTensorf);

        CHECK((accumulatorTest(srcTensori, dstTensori, binop::minus)));
        CHECK((accumulatorTest(srcTensorf, dstTensorf, binop::minus)));
    }
}

TEST_CASE_TEMPLATE("Test:Conv2D channels-first", Device,
#ifdef BACKEND_CUDNN
    upstride::device::CUDA,
#endif
    upstride::device::CPU)
{
    static upstride::Context& context(getContextInstance<Device>());
    static Device device(context);

    static const bool isCpu = std::is_same<Device, upstride::device::CPU>();
    std::cout << " -- NCHW Conv2D on " << (isCpu ? "CPU" : "GPU") << "..." << std::endl;

    SUBCASE("Conv2D - complex") {
        // convolving a 3x2x2 CHW complex tensor with a 1x1 convolution kernel with 2 channels on output
        std::cout << "    Conv2D - complex" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<Device, float> input(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 3, 2, 2});
        input = {
            // real part
            0, 0, 0, 0,
            -1, 0, 0, 0,
            1, 2, 3, 4,

            // imaginary part
            1, 1, 1, 1,
            0, 0, 0, 0,
            -1, -2, -3, -4
        };

        // set up filter tensor
        AllocatedTensor<Device, float> kernel(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 3, 1, 1});
        kernel = {
            // real part
            1, 1, 1,
            0, 0, 2,

            // imaginary part
            0, 0, 0,
            -1, 0, 0
        };

        // set up reference output
        AllocatedTensor<Device, float> refOutput(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 2, 2});
        refOutput = {
           // real part
           0, 2, 3, 4,
           2+1, 4+1, 6+1, 8+1,

           // imaginary part
           0, -1, -2, -3,
           -2, -4, -6, -8
        };

        // compute test output
        AllocatedTensor<Device, float> testOutput(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 2, 2});

        upstride::Conv2DFwdDescriptor descriptor(input.getShape(),
                                              kernel.getShape(),
                                              1,                        //stride
                                              1,                        //dilation
                                              upstride::Padding::VALID, //padding preset
                                              {},                       //explicit padding
                                              1,                        //groups
                                              Algebra::COMPLEX,         //algebra
                                              DataFormat::NCHW,         //dataformat
                                              FilterLayout::OIHW,       //filter layout
                                              false);                   //use bias
        upstride::conv2DFwd<Device, float>(device, device,
                                                input,
                                                kernel,
                                                nullptr,
                                                testOutput,
                                                descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("Conv2D - quaternion") {
        using namespace upstride;

        const int dim = MULTIVECTOR_DIM[Algebra::QUATERNION];
        const int N = 1, C = 2, H = 3, W = 3;
        const int numel = dim * N * C * H * W;

        std::cout << "    Conv2D - quaternion" << std::endl;

        AllocatedTensor<Device, float> input(device, {dim*N, C, H, W});
        AllocatedTensor<Device, float> kernel(device, {dim, N, C, H, W});
        AllocatedTensor<Device, float> output(device, {dim*N, 1, 1, 1});

        const std::vector<float> ones(numel, 1.0f);
        input = ones;
        kernel = ones;

        Conv2DFwdDescriptor descriptor(input.getShape(), kernel.getShape(), 1, 1, Padding::VALID, {}, 1, Algebra::QUATERNION, DataFormat::NCHW, FilterLayout::OIHW, false);
        conv2DFwd<Device, float>(device, device, input, kernel, nullptr, output, descriptor);

        std::vector<float> out = output;
        CHECK((out.size() == 4));
        CHECK((out[0] == -36.0f && out[1] == 36.0f && out[2] == 36.0f && out[3] == 36.0f));
    }

    SUBCASE("Conv2D - quaternion with real input") {
        // convolving a 3x2x2 CHW real tensor with a 1x1 convolution kernel with 2 channels on output
        std::cout << "    Conv2D - quaternion with real input" << std::endl;
        using namespace upstride;

        // set up a real input tensor
        AllocatedTensor<Device, float> input(device, {1, 3, 2, 2});
        input = {
            0, 0, 0, 0,
            -1, 0, 0, 0,
            1, 2, 3, 4,
        };

        // set up a quaternion filter tensor
        AllocatedTensor<Device, float> kernel(device, {MULTIVECTOR_DIM[Algebra::QUATERNION], 2, 3, 1, 1});
        kernel = {
            1, 1, 1,
            0, 0, 2,

            0, 0, 0,
            -1, -1, -1,

            0, 1, 2,
            0, 2, 4,

            -1, -2, -3,
            0, 0, 1
        };

        // set up a reference output
        AllocatedTensor<Device, float> refOutput(device, {MULTIVECTOR_DIM[Algebra::QUATERNION], 2, 2, 2});
        refOutput = {
            0, 2, 3, 4,
            2, 4, 6, 8,

            0, 0, 0, 0,
            0, -2, -3, -4,

            1, 4, 6, 8,
            2, 8, 12, 16,

            -1, -6, -9, -12,
            1, 2, 3, 4
        };

        // compute test output
        AllocatedTensor<Device, float> testOutput(device, {MULTIVECTOR_DIM[Algebra::QUATERNION], 2, 2, 2});
        Conv2DFwdDescriptor descriptor(input.getShape(), kernel.getShape(), 1, 1, Padding::VALID, {}, 1, Algebra::QUATERNION, DataFormat::NCHW, FilterLayout::OIHW, false, true);
        conv2DFwd<Device, float>(device, device, input, kernel, nullptr, testOutput, descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("Conv2D - GA(3,0,0)") {
        // convolving a 1x1x1 CHW GA(3,0,0) tensor with a 1x1 convolution kernel with 2 channels on output
        // cheating: using quaternion dimensions subset
        std::cout << "    Conv2D - GA(3,0,0)" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<Device, float> input(device, {MULTIVECTOR_DIM[Algebra::GA_300], 1, 1, 1});
        input = {
            1,      // r
            0,      // e1
            0,      // e2
            0,      // e3
            2,      // e12
            4,      // e13
            3,      // e23
            0,      // e123
        };

        // set up filter tensor
        AllocatedTensor<Device, float> kernel(device, {MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1, 1});
        kernel = {
            -1,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -2,   0,   // e12
            -4,   0,   // e13
            -3,   0,   // e23
             0,   0   // e123
        };

        // set up reference output
        AllocatedTensor<Device, float> refOutput(device, {MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1});
        refOutput = {
            28,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -4,  20,   // e12
            -8,  40,   // e13
            -6,  30,   // e23
             0,   0    // e123
        };

        // compute test output
        AllocatedTensor<Device, float> testOutput(device, {MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1});
        Conv2DFwdDescriptor descriptor(input.getShape(), kernel.getShape(), 1, 1, Padding::VALID, {}, 1, Algebra::GA_300, DataFormat::NCHW, FilterLayout::OIHW, false);
        conv2DFwd<Device, float>(device, device, input, kernel, nullptr, testOutput, descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }
}



TEST_CASE_TEMPLATE("Test:Conv2D channels-last", Device,
#ifdef BACKEND_CUDNN
    upstride::device::CUDA,
#endif
    upstride::device::CPU)
{
    static upstride::Context& context(getContextInstance<Device>());
    static Device device(context);

    static const bool isCpu = std::is_same<Device, upstride::device::CPU>();
    std::cout << " -- NHWC Conv2D on " << (isCpu ? "CPU" : "GPU") << "..." << std::endl;

    SUBCASE("Conv2D real") {
        // convolving a 3x2x2 CHW real tensor with a 1x1 convolution kernel with 2 channels on output
        std::cout << "    Conv2D - real" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<Device, float> input(device, {1, 2, 2, 3});
        input = { 0, -1,  1,
                  0,  0,  2,
                  0,  0,  3,
                  1,  0,  4 };

        // set up filter tensor
        AllocatedTensor<Device, float> kernel(device, {2, 1, 1, 3});
        kernel = { 1, 1, 1,
                   0, 0, 2 };

        // set up the expected output
        AllocatedTensor<Device, float> refOutput(device, {1, 2, 2, 2});
        refOutput = { 0, 2,
                      2, 4,
                      3, 6,
                      5, 8 };

        // compute the test output
        AllocatedTensor<Device, float> testOutput(device, {1, 2, 2, 2});

        upstride::Conv2DFwdDescriptor descriptor(input.getShape(),
                                                 kernel.getShape(),
                                                 1,                        //stride
                                                 1,                        //dilation
                                                 upstride::Padding::VALID, //padding preset
                                                 {},                       //explicit padding
                                                 1,                        //groups
                                                 Algebra::REAL,            //algebra
                                                 DataFormat::NHWC,         //dataformat
                                                 FilterLayout::OHWI,       //filter layout
                                                 false);                   //use bias
        upstride::conv2DFwd<Device, float>(device, device,
                                           input,
                                           kernel,
                                           nullptr,
                                           testOutput,
                                           descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }
}


TEST_CASE_TEMPLATE("Test:Dense", Device,
#ifdef BACKEND_CUDNN
    upstride::device::CUDA,
#endif
    upstride::device::CPU)
{
    static upstride::Context& context(getContextInstance<Device>());
    static Device device(context);

    static const bool isCpu = std::is_same<Device, upstride::device::CPU>();
    std::cout << " -- Dense on " << (isCpu ? "CPU" : "GPU") << "..." << std::endl;

    SUBCASE("Dense - complex") {
        std::cout << "    Dense - complex" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<Device, float> input(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 3});
        input = {
            +1, 0, 2,
            0, -1, 2
        };

        // set up filter tensor
        AllocatedTensor<Device, float> filter(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 3, 2});
        filter = {
            // real part
            1, 0,
            0, 0,
            0, 10,

            // imaginary part
            0, 0,
            0, 1,
            0, 10
        };

        // set up reference output
        AllocatedTensor<Device, float> refOutput(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 2});
        refOutput = {
           1, 20+1-20,
           0, 20+20
        };

        // allocate output
        AllocatedTensor<Device, float> testOutput(device, {MULTIVECTOR_DIM[Algebra::COMPLEX], 2});

        // compute test output
        const DenseFwdDescriptor descriptor(input.getShape(), filter.getShape(), Algebra::COMPLEX, FilterLayout::IO, false);
        upstride::denseFwd<Device, float>(device, device, input, filter, nullptr, testOutput, descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("Dense - GA(3,0,0)") {
        std::cout << "    Dense - GA(3,0,0)" << std::endl;
        using namespace upstride;

        // reusing data from Conv2D test; set up input tensor
        AllocatedTensor<Device, float> input(device, {MULTIVECTOR_DIM[Algebra::GA_300], 1});
        input = {1, 0, 0, 0, 2, 4, 3, 0, };

        // set up filter tensor
        AllocatedTensor<Device, float> filter(device, {MULTIVECTOR_DIM[Algebra::GA_300], 1, 2});
        filter = {
            -1,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -2,   0,   // e12
            -4,   0,   // e13
            -3,   0,   // e23
             0,   0    // e123
        };

        // set up reference output
        AllocatedTensor<Device, float> refOutput(device, {MULTIVECTOR_DIM[Algebra::GA_300], 2});
        refOutput = {
            28,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -4,  20,   // e12
            -8,  40,   // e13
            -6,  30,   // e23
             0,   0    // e123
        };

        // allocate test output
        AllocatedTensor<Device, float> testOutput(device, {MULTIVECTOR_DIM[Algebra::GA_300], 2});

        // compute test output
        const DenseFwdDescriptor descriptor(input.getShape(), filter.getShape(), Algebra::GA_300, FilterLayout::IO, false);
        upstride::denseFwd<Device, float>(device, device, input, filter, nullptr, testOutput, descriptor);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }
}

TEST_CASE("Test:DataFormat") {
    std::cout << " -- DataFormat functions..." << std::endl;

    SUBCASE("DataFormat - dataFormatFromString") {
        std::cout << "    DataFormat - dataFormatFromString" << std::endl;

        upstride::DataFormat dfNCHW = upstride::dataFormatFromString("NCHW");
        upstride::DataFormat dfNHWC = upstride::dataFormatFromString("NHWC");

        CHECK((dfNCHW == upstride::DataFormat::NCHW));
        CHECK((dfNHWC == upstride::DataFormat::NHWC));
    }

    SUBCASE("DataFormat - get{W,H,D}DimensionNumber") {
        std::cout << "    DataFormat - get{W,H,D}DimensionNumber" << std::endl;

        CHECK((3 == upstride::getWidthDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((2 == upstride::getWidthDimensionNumber(upstride::DataFormat::NHWC)));

        CHECK((2 == upstride::getHeightDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((1 == upstride::getHeightDimensionNumber(upstride::DataFormat::NHWC)));

        CHECK((1 == upstride::getDepthDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((3 == upstride::getDepthDimensionNumber(upstride::DataFormat::NHWC)));
    }
}

TEST_CASE("Test:Padding") {
    std::cout << " -- Padding functions..." << std::endl;

    SUBCASE("Padding - paddingFromString") {
        std::cout << "    Padding - paddingFromString" << std::endl;

        upstride::Padding padSame = upstride::paddingFromString("SAME");
        upstride::Padding padValid = upstride::paddingFromString("VALID");
        upstride::Padding padExplicit = upstride::paddingFromString("EXPLICIT");

        CHECK((padSame == upstride::Padding::SAME));
        CHECK((padValid == upstride::Padding::VALID));
        CHECK((padExplicit == upstride::Padding::EXPLICIT));
    }
}

TEST_CASE("Test:Utils") {
    std::cout << " -- Utils tests..." << std::endl;

    SUBCASE("Utils::computeConvOutputSize") {
        std::cout << "    Utils::computeConvOutputSize" << std::endl;

        const upstride::Shape inputShape({1, 224, 224, 3});
        const upstride::Shape kernelShape({4, 32, 3, 3, 2});
        const upstride::Shape expectedShape({1, 224, 224, 32});

        upstride::Padding paddingPreset = upstride::Padding::SAME;
        const std::vector<int32_t>& explicitPadding = {0, 0};
        const std::vector<int32_t>& stride = {1, 1};
        const std::vector<int32_t>& dilation = {0, 0};
        upstride::IntPair padBefore, padAfter;

        upstride::Conv2DFwdDescriptor descriptor(inputShape,
                                                 kernelShape,
                                                 stride,
                                                 dilation,
                                                 paddingPreset,
                                                 explicitPadding,
                                                 1,
                                                 upstride::Algebra::QUATERNION,
                                                 upstride::DataFormat::NHWC,
                                                 upstride::FilterLayout::OHWI,
                                                 false);


        upstride::Shape outputShape = descriptor.getOutputShape();

        CHECK((outputShape == expectedShape));
    }
}

TEST_CASE("Test:TensorSplit") {
    static upstride::onednn::Context context;
    static upstride::device::CPU device(context);

    static const int TEST_BATCH_SIZE = 4;
    static const int TEST_DATA[TEST_BATCH_SIZE * 2] = {1, 1, 2, 2, 3, 3, 4, 4};
    static const upstride::Shape TEST_DATA_SHAPE{4, 2, 1, 1};

    upstride::Tensor<upstride::device::CPU, const int> testInputTensor(device, TEST_DATA_SHAPE, TEST_DATA);

    SUBCASE("TensorSplit keeping outermost dimension") {
        upstride::TensorSplit<upstride::device::CPU, const int, TEST_BATCH_SIZE> split(testInputTensor, true);
        static const upstride::Shape EXPECTED_PART_SHAPE{1, 2, 1, 1};

        for (unsigned i = 0; i < TEST_BATCH_SIZE; ++i) {
            CHECK(split[i].getShape() == EXPECTED_PART_SHAPE);
            const int* ptr = split[i].getDataPtr();
            CHECK(*ptr == i + 1);
        }
    }

    SUBCASE("TensorSplit without keeping the outermost dimension") {
        upstride::TensorSplit<upstride::device::CPU, const int, TEST_BATCH_SIZE> split(testInputTensor, false);
        static const upstride::Shape EXPECTED_PART_SHAPE{2, 1, 1};

        for (unsigned i = 0; i < TEST_BATCH_SIZE; ++i) {
            CHECK(split[i].getShape() == EXPECTED_PART_SHAPE);
            const int* ptr = split[i].getDataPtr();
            CHECK(*ptr == i + 1);
        }
    }
}

TEST_CASE("Algebras multivector dimensions") {
    CHECK(upstride::MULTIVECTOR_DIM[upstride::Algebra::REAL] == 1);
    CHECK(upstride::MULTIVECTOR_DIM[upstride::Algebra::COMPLEX] == 2);
    CHECK(upstride::MULTIVECTOR_DIM[upstride::Algebra::QUATERNION] == 4);
    CHECK(upstride::MULTIVECTOR_DIM[upstride::Algebra::GA_300] == 8);
}

#ifdef BACKEND_CUDNN
TEST_CASE("Crop / insert GPU kernels") {
    std::cout << " -- Crop / insert GPU kernels..." << std::endl;
    using namespace upstride;
    static cudnn::Context context;
    static device::CUDA gpu(context);

    SUBCASE("NCHW crop kernel test") {
        std::cout << "    Channels-first crop" << std::endl;
        AllocatedTensor<device::CUDA, float> input(gpu, {1, 2, 5, 7});
        input = { 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 3, 5, 0, 0,
                  0, 0, 4, 6, 8, 0, 0,
                  0, 0, 3, 1, 9, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,

                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 9, 2, 3, 0, 0,
                  0, 0, 8, 1, 4, 0, 0,
                  0, 0, 7, 6, 5, 0, 0,
                  0, 0, 0, 0, 0, 0, 0 };

        AllocatedTensor<device::CUDA, float> testOutput(gpu, {1, 2, 3, 3});
        cudnn::crop(input, testOutput, DataFormat::NCHW, {1, 2});

        AllocatedTensor<device::CUDA, float> refOutput(gpu, {1, 2, 3, 3});
        refOutput = { 1, 3, 5,
                      4, 6, 8,
                      3, 1, 9,

                      9, 2, 3,
                      8, 1, 4,
                      7, 6, 5 };

        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("NCHW insert kernel test") {
        std::cout << "    Channels-first insert" << std::endl;
        AllocatedTensor<device::CUDA, float> input(gpu, {1, 2, 3, 3});
        input = { 1, 3, 5,
                  4, 6, 8,
                  3, 1, 9,

                  9, 2, 3,
                  8, 1, 4,
                  7, 6, 5 };

        AllocatedTensor<device::CUDA, float> testOutput(gpu, {1, 2, 5, 7});
        testOutput.zero();
        cudnn::insert(Tensor<device::CUDA, const float>(gpu, input.getShape(), input.getDataPtr()),
                      testOutput,
                      DataFormat::NCHW,
                      {1, 2});
        // FIXME: input scalar datatype of cudnn::insert is const-qualified

        AllocatedTensor<device::CUDA, float> refOutput(gpu, {1, 2, 5, 7});
        refOutput = { 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 3, 5, 0, 0,
                      0, 0, 4, 6, 8, 0, 0,
                      0, 0, 3, 1, 9, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,

                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 9, 2, 3, 0, 0,
                      0, 0, 8, 1, 4, 0, 0,
                      0, 0, 7, 6, 5, 0, 0,
                      0, 0, 0, 0, 0, 0, 0 };

        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("NHWC crop kernel test") {
        std::cout << "    Channels-last crop" << std::endl;
        AllocatedTensor<device::CUDA, float> input(gpu, {1, 3, 4, 2});
        input = { 0,0, 0,0, 0,0, 0,0,
                  0,0, 0,0, 1,2, 3,4,
                  0,0, 0,0, 5,6, 7,8 };

        AllocatedTensor<device::CUDA, float> refOutput(gpu, {1, 2, 2, 2});
        refOutput = { 1,2, 3,4,
                      5,6, 7,8 };

        AllocatedTensor<device::CUDA, float> testOutput(gpu, refOutput.getShape());
        cudnn::crop(input, testOutput, DataFormat::NHWC, {1, 2});

        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE("NHWC insert kernel test") {
        std::cout << "    Channels-last insert" << std::endl;
        AllocatedTensor<device::CUDA, float> input(gpu, {1, 2, 2, 2});
        input = { 1,2, 3,4,
                  5,6, 7,8 };

        AllocatedTensor<device::CUDA, float> refOutput(gpu, {1, 3, 4, 2});
        refOutput = { 0,0, 0,0, 0,0, 0,0,
                      0,0, 0,0, 1,2, 3,4,
                      0,0, 0,0, 5,6, 7,8 };

        AllocatedTensor<device::CUDA, float> testOutput(gpu, refOutput.getShape());
        cudnn::insert(Tensor<device::CUDA, const float>(gpu, input.getShape(), input.getDataPtr()),
                      testOutput, DataFormat::NHWC,
                      {1, 2});

        CHECK(compareTensors(refOutput, testOutput));
    }

}
#endif