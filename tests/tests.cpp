/**
                                                    Report

    The following code contains the unit tests for the c++ part of the phoenix engine, it performs
    basic operations verifications and verifies the maths properties for each operator.

    It contains:
            TODO: write all tested functionalities
**/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <stdlib.h>  // calloc
// #include <fstream>
#include <iostream>  // cout
// #include <limits>
#include <random>

#include "doctest/doctest.h"
#include "upstride.hpp"

static upstride::device::CPU device;
static upstride::onednn::Context context;

/**
 * @brief Fill a tensor with random floating values.
 * 
 * @param t Tensor to fill
 */
void setRandVal(upstride::AllocatedTensor<upstride::device::CPU, float>& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);
    for (int i = 0; i < t.getShape().numel(); i++) {
        t.getDataPtr()[i] = dist(gen);
    }
}

void setRandVal(upstride::AllocatedTensor<upstride::device::CPU, int>& t) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-10, 10);
    for (int i = 0; i < t.getShape().numel(); i++) {
        t.getDataPtr()[i] = dist(gen);
    }
}

/**
 * @brief Fills a tensor with a set of values
 * @tparam T        Tensor scalar datatype
 * @param tensor    The tensor to fill
 * @param values    The values
 */
template<typename T>
void assign(upstride::Tensor<upstride::device::CPU, T>& tensor, std::initializer_list<T> values) {
    if (values.size() != tensor.getShape().numel())
        throw std::invalid_argument("Number of input values does not match the tensor size");
    T* ptr = tensor.getDataPtr();
    for (float val : values) {
        *ptr = val;
        ++ptr;
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
template <typename T>
bool compareTensors(const upstride::Tensor<upstride::device::CPU, T>& lhs, upstride::Tensor<upstride::device::CPU, T>& rhs, const T threshold = (T)0) {
    if (lhs.getShape() != rhs.getShape())
        return false;
    const T *l = lhs.getDataPtr(), *r = rhs.getDataPtr();
    for (int i = 0; i < lhs.getShape().numel(); ++i)
        if (std::abs(l[i] - r[i]) > threshold)
            return false;
    return true;
}


TEST_CASE("Test:Shape") {
    std::cout << "---- Test: Shape creation" << std::endl;

    SUBCASE(" Test: Shape::Shape()") {
        std::cout << " Test: Shape::Shape()" << std::endl;
        upstride::Shape s1({1, 2, 3, 4});

        std::cout << s1 << std::endl;

        CHECK((s1.getShapePtr()[0] == 1));
        CHECK((s1.getShapePtr()[1] == 2));
        CHECK((s1.getShapePtr()[2] == 3));
        CHECK((s1.getShapePtr()[3] == 4));
        std::cout << std::endl;
    }

    SUBCASE(" Test: Shape == Shape") {
        std::cout << " Test: Shape::operator==" << std::endl;
        const upstride::Shape s1({1, 2, 3, 4});
        const upstride::Shape s2({1, 2, 3, 4});

        CHECK((s1 == s2));
        std::cout << std::endl;
    }

    SUBCASE(" Test: Shape != Shape") {
        std::cout << " Test: Shape::operator!=" << std::endl;
        const upstride::Shape s1({1, 2, 3, 4});
        const upstride::Shape s2({1, 2, 6, 4});
        const upstride::Shape s3({1, 2});

        CHECK((s1 != s2));
        CHECK((s1 != s3));
        std::cout << std::endl;
    }

    SUBCASE(" Test: Shape slicing and splitting") {
        std::cout << " Shape slicing and splitting" << std::endl;
        const upstride::Shape testShape{0, 1, 2, 3, 4, 5, 6};

        CHECK((testShape.slice(0, 3) == upstride::Shape{0, 1, 2}));
        CHECK((testShape.slice(4) == upstride::Shape{4, 5, 6}));
        CHECK((testShape.slice(-3) == upstride::Shape{4, 5, 6}));
        CHECK((testShape.slice(-3, -1) == upstride::Shape{4, 5}));

        CHECK((testShape.slice(2, 3).split(1) == upstride::Shape{2}));
        CHECK((testShape.slice(3).split(3) == upstride::Shape{1, 4, 5, 6}));
        CHECK((testShape.slice(4).split(2) == upstride::Shape{2, 5, 6}));

        std::cout << std::endl;
    }
}

enum binop { plus,
             minus };
template <typename T>
bool accumulatorTest(const upstride::AllocatedTensor<upstride::device::CPU, T>& srcTensor, upstride::AllocatedTensor<upstride::device::CPU, T>& dstTensor, binop op) {
    upstride::AllocatedTensor<upstride::device::CPU, T> dstCopyTensor(device, dstTensor.getShape());

    for (int i = 0; i < dstTensor.getShape().numel(); i++) {
        dstCopyTensor.getDataPtr()[i] = dstTensor.getDataPtr()[i];
    }
    if (op == binop::plus) {
        dstTensor += srcTensor;
        for (int i = 0; i < dstTensor.getShape().numel(); i++) {
            if (dstTensor.getDataPtr()[i] != dstCopyTensor.getDataPtr()[i] + srcTensor.getDataPtr()[i]) {
                return false;
            }
        }
    } else if (op == binop::minus) {
        dstTensor -= srcTensor;
        for (int i = 0; i < dstTensor.getShape().numel(); i++) {
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
    std::cout << "---- Test: Tensor creation" << std::endl;
    int H = 224, W = 224, C = 3;
    int numel = H * W * C;
    upstride::Shape s1({1, C, H, W});

    SUBCASE(" Test: Tensor::Tensor()") {
        std::cout << " Test: Tensor::Tensor()" << std::endl;

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
        std::cout << std::endl;
    }

    SUBCASE(" Test: [float/int] Tensor Src + Dst") {
        std::cout << "[float] Test: Tensor Src + Dst" << std::endl;
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(device, s1), dstTensori(device, s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(device, s1), dstTensorf(device, s1);
        setRandVal(srcTensori);
        setRandVal(dstTensori);
        setRandVal(srcTensorf);
        setRandVal(dstTensorf);

        CHECK((accumulatorTest(srcTensori, dstTensori, binop::plus)));
        CHECK((accumulatorTest(srcTensorf, dstTensorf, binop::plus)));
        std::cout << std::endl;
    }

    SUBCASE(" Test: [float/int] Tensor Src - Dst") {
        std::cout << "[float/int] Test: Tensor Src - Dst" << std::endl;
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(device, s1), dstTensori(device, s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(device, s1), dstTensorf(device, s1);
        setRandVal(srcTensori);
        setRandVal(dstTensori);
        setRandVal(srcTensorf);
        setRandVal(dstTensorf);

        CHECK((accumulatorTest(srcTensori, dstTensori, binop::minus)));
        CHECK((accumulatorTest(srcTensorf, dstTensorf, binop::minus)));
        std::cout << std::endl;
    }
}

TEST_CASE("Test:Conv2d") {
    std::cout << "---- Test: Conv2d computation" << std::endl;

    SUBCASE(" Test: Conv2d - complex") {
        // convolving a 3x2x2 CHW complex tensor with a 1x1 convolution kernel with 2 channels on output
        std::cout << " Test: Conv2d - complex" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<device::CPU, float> input(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 3, 2, 2}));
        assign<float>(input, {
            // real part
            0, 0, 0, 0,
            -1, 0, 0, 0,
            1, 2, 3, 4,

            // imaginary part
            1, 1, 1, 1,
            0, 0, 0, 0,
            -1, -2, -3, -4
        });

        // set up filter tensor
        AllocatedTensor<device::CPU, float> kernel(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 3, 1, 1}));
        assign<float>(kernel, {
            // real part
            1, 1, 1,
            0, 0, 2,

            // imaginary part
            0, 0, 0,
            -1, 0, 0
        });

        // set up reference output
        AllocatedTensor<device::CPU, float> refOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 2, 2}));
        assign<float>(refOutput, {
           // real part
           0, 2, 3, 4,
           2+1, 4+1, 6+1, 8+1,

           // imaginary part
           0, -1, -2, -3,
           -2, -4, -6, -8
        });

        // init operation
        upstride::UpstrideConv2DFunctor<upstride::device::CPU, float> op(context, Algebra::COMPLEX, DataFormat::NCHW, 1, 1, false);

        // compute test output
        AllocatedTensor<device::CPU, float> testOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 2, 2, 2}));
        op(::device, input, kernel, nullptr, testOutput, 0, 0);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE(" Test: Conv2d - quaternion") {
        const int dim = upstride::MULTIVECTOR_DIM[upstride::Algebra::QUATERNION];
        const int N = 1, C = 2, H = 3, W = 3;
        const int numel = dim * N * C * H * W;

        std::cout << " Test: Conv2d - quaternion" << std::endl;
        const upstride::Algebra algebra(upstride::Algebra::QUATERNION);

        upstride::Shape sIn({dim*N, C, H, W});
        upstride::Shape sKer({dim, N, C, H, W});
        upstride::Shape sBias({});
        upstride::Shape sOut({dim*N, 1, 1, 1});
        upstride::AllocatedTensor<upstride::device::CPU, float> inputTensor(device, sIn);
        upstride::AllocatedTensor<upstride::device::CPU, float> kernelTensor(device, sKer);
        upstride::AllocatedTensor<upstride::device::CPU, float> outputTensor(device, sOut);
        outputTensor.zero();

        float* inputTensorPtr = inputTensor.getDataPtr();
        float* kernelTensorPtr = kernelTensor.getDataPtr();
        for(int i = 0; i < numel; ++i) {
            inputTensorPtr[i] = 1;
            kernelTensorPtr[i] = 1;
        }

        upstride::IntPair st(1, 1);
        upstride::IntPair dil(1, 1);
        const upstride::IntPair padBefore(0);
        const upstride::IntPair padAfter(0);
        
        upstride::UpstrideConv2DFunctor<upstride::device::CPU, float> myConv2DFunctor(context, algebra, upstride::DataFormat::NCHW, st, dil, false);
        myConv2DFunctor(device, inputTensor, kernelTensor, nullptr, outputTensor, padBefore, padAfter, /*groups=*/1);

        bool test = true;
        float* outputTensorPtr = outputTensor.getDataPtr();
        if (outputTensorPtr[0] != -36.0f)
            test = false;
        for (int i = 1; i < 4 && test; ++i) {
            if (outputTensorPtr[i] != 36.0f)
                test = false;
        }
        CHECK((test));
    }

    SUBCASE(" Test: Conv2d - GA(3,0,0)") {
        // convolving a 1x1x1 CHW GA(3,0,0) tensor with a 1x1 convolution kernel with 2 channels on output
        // cheating: using quaternion dimensions subset
        std::cout << " Test: Conv2d - GA(3,0,0)" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<device::CPU, float> input(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 1, 1, 1}));
        assign<float>(input, {
            1,      // r
            0,      // e1
            0,      // e2
            0,      // e3
            2,      // e12
            4,      // e13
            3,      // e23
            0,      // e123
        });

        // set up filter tensor
        AllocatedTensor<device::CPU, float> kernel(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1, 1}));
        assign<float>(kernel, {
            -1,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -2,   0,   // e12
            -4,   0,   // e13
            -3,   0,   // e23
             0,   0   // e123
        });

        // set up reference output
        AllocatedTensor<device::CPU, float> refOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1}));
        assign<float>(refOutput, {
            28,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -4,  20,   // e12
            -8,  40,   // e13
            -6,  30,   // e23
             0,   0    // e123
        });

        // init operation
        upstride::UpstrideConv2DFunctor<upstride::device::CPU, float> op(context, Algebra::GA_300, DataFormat::NCHW, 1, 1, false);

        // compute test output
        AllocatedTensor<device::CPU, float> testOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 2, 1, 1}));
        op(::device, input, kernel, nullptr, testOutput, 0, 0);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }
}


TEST_CASE("Test:Dense") {
    std::cout << "---- Test: Dense computation" << std::endl;

    SUBCASE(" Test: Dense - complex") {
        std::cout << " Test: Dense - complex" << std::endl;
        using namespace upstride;

        // set up input tensor
        AllocatedTensor<device::CPU, float> input(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 3}));
        assign<float>(input, {
            +1, 0, 2,
            0, -1, 2
        });

        // set up filter tensor
        AllocatedTensor<device::CPU, float> kernel(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 3, 2}));
        assign<float>(kernel, {
            // real part
            1, 0,
            0, 0,
            0, 10,

            // imaginary part
            0, 0,
            0, 1,
            0, 10
        });

        // set up reference output
        AllocatedTensor<device::CPU, float> refOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 2}));
        assign<float>(refOutput, {
           1, 20+1-20,
           0, 20+20
        });

        // init operation
        upstride::UpstrideDenseFunctor<upstride::device::CPU, float> op(
            context,
            Algebra::COMPLEX,
            DataFormat::NC,
            false
        );

        // compute test output
        AllocatedTensor<device::CPU, float> testOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::COMPLEX], 2}));
        op(::device, input, kernel, nullptr, testOutput);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }

    SUBCASE(" Test: Dense - GA(3,0,0)") {
        std::cout << " Test: Dense - GA(3,0,0)" << std::endl;
        using namespace upstride;

        // reusing data from Conv2D test; set up input tensor
        AllocatedTensor<device::CPU, float> input(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 1}));
        assign<float>(input, {1, 0, 0, 0, 2, 4, 3, 0, });

        // set up filter tensor
        AllocatedTensor<device::CPU, float> kernel(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 1, 2}));
        assign<float>(kernel, {
            -1,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -2,   0,   // e12
            -4,   0,   // e13
            -3,   0,   // e23
             0,   0   // e123
        });

        // set up reference output
        AllocatedTensor<device::CPU, float> refOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 2}));
        assign<float>(refOutput, {
            28,  10,   // r
             0,   0,   // e1
             0,   0,   // e2
             0,   0,   // e3
            -4,  20,   // e12
            -8,  40,   // e13
            -6,  30,   // e23
             0,   0    // e123
        });

        // init operation
        upstride::UpstrideDenseFunctor<upstride::device::CPU, float> op(
            context,
            Algebra::GA_300,
            DataFormat::NC,
            false
        );

        // compute test output
        AllocatedTensor<device::CPU, float> testOutput(::device, Shape({MULTIVECTOR_DIM[Algebra::GA_300], 2}));
        op(::device, input, kernel, nullptr, testOutput);

        // compare
        CHECK(compareTensors(refOutput, testOutput));
    }
}

TEST_CASE("Test:DataFormat") {
    std::cout << "---- Test: DataFormat functions" << std::endl;

    SUBCASE(" Test: DataFormat - dataFormatFromString") {
        std::cout << " Test: DataFormat - dataFormatFromString" << std::endl;

        upstride::DataFormat dfNCHW = upstride::dataFormatFromString("NCHW");
        upstride::DataFormat dfNHWC = upstride::dataFormatFromString("NHWC");

        CHECK((dfNCHW == upstride::DataFormat::NCHW));
        CHECK((dfNHWC == upstride::DataFormat::NHWC));
        std::cout << std::endl;
    }

    SUBCASE(" Test: DataFormat - get{W,H,D}DimensionNumber") {
        std::cout << " Test: DataFormat - get{W,H,D}DimensionNumber" << std::endl;

        CHECK((3 == upstride::getWidthDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((2 == upstride::getWidthDimensionNumber(upstride::DataFormat::NHWC)));

        CHECK((2 == upstride::getHeightDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((1 == upstride::getHeightDimensionNumber(upstride::DataFormat::NHWC)));

        CHECK((1 == upstride::getDepthDimensionNumber(upstride::DataFormat::NCHW)));
        CHECK((3 == upstride::getDepthDimensionNumber(upstride::DataFormat::NHWC)));
        std::cout << std::endl;
    }
}

TEST_CASE("Test:Padding") {
    std::cout << "---- Test: Padding functions" << std::endl;

    SUBCASE(" Test: Padding - paddingFromString") {
        std::cout << " Test: Padding - paddingFromString" << std::endl;

        upstride::Padding padSame = upstride::paddingFromString("SAME");
        upstride::Padding padValid = upstride::paddingFromString("VALID");
        upstride::Padding padExplicit = upstride::paddingFromString("EXPLICIT");

        CHECK((padSame == upstride::Padding::SAME));
        CHECK((padValid == upstride::Padding::VALID));
        CHECK((padExplicit == upstride::Padding::EXPLICIT));
        std::cout << std::endl;
    }
}

TEST_CASE("Test:Utils") {
    std::cout << "---- Test: Utils functions" << std::endl;

    SUBCASE(" Test: Utils::computeConvOutputSize") {
        std::cout << " Test: Utils::computeConvOutputSize" << std::endl;

        const upstride::DataFormat df = upstride::DataFormat::NCHW;

        const upstride::Shape inputShape({1, 224, 224, 3});
        const upstride::Shape kernelShape({4, 3, 3, 3, 32});
        const upstride::Shape expectedShape({1, 3, 224, 3});

        upstride::Padding paddingPreset = upstride::Padding::SAME;
        const std::vector<int32_t>& explicitPadding = {0, 0};
        const std::vector<int32_t>& stride = {1, 1};
        const std::vector<int32_t>& dilation = {0, 0};
        upstride::IntPair padBefore, padAfter;

        upstride::Shape outputShape = upstride::computeConvOutputSize(upstride::Algebra::QUATERNION,
                                                                      df,
                                                                      inputShape,
                                                                      kernelShape,
                                                                      paddingPreset,
                                                                      explicitPadding,
                                                                      stride,
                                                                      dilation,
                                                                      padBefore, padAfter);

        std::cout << outputShape << std::endl;

        CHECK((outputShape == expectedShape));
        std::cout << std::endl;
    }
}

TEST_CASE("Test:TensorSplit") {
    static const int TEST_BATCH_SIZE = 4;
    static const int TEST_DATA[TEST_BATCH_SIZE * 2] = {1, 1, 2, 2, 3, 3, 4, 4};
    static const upstride::Shape TEST_DATA_SHAPE{4, 2, 1, 1};

    upstride::Tensor<upstride::device::CPU, const int> testInputTensor(device, TEST_DATA_SHAPE, TEST_DATA);

    SUBCASE(" Test: TensorSplit keeping outermost dimension") {
        upstride::TensorSplit<upstride::device::CPU, const int, TEST_BATCH_SIZE> split(testInputTensor, true);
        static const upstride::Shape EXPECTED_PART_SHAPE{1, 2, 1, 1};

        for (int i = 0; i < TEST_BATCH_SIZE; ++i) {
            CHECK(split[i].getShape() == EXPECTED_PART_SHAPE);
            const int* ptr = split[i].getDataPtr();
            CHECK(*ptr == i + 1);
        }
    }

    SUBCASE(" Test: TensorSplit without keeping the outermost dimension") {
        upstride::TensorSplit<upstride::device::CPU, const int, TEST_BATCH_SIZE> split(testInputTensor, false);
        static const upstride::Shape EXPECTED_PART_SHAPE{2, 1, 1};

        for (int i = 0; i < TEST_BATCH_SIZE; ++i) {
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
