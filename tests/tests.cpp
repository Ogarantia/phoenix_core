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

/* =============================================================================
                                 PHOENIX 
============================================================================= */
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

    SUBCASE(" Test: Shape !(==) Shape") {
        std::cout << " Test: !(Shape::operator==)" << std::endl;
        const upstride::Shape s1({1, 2, 3, 4});
        const upstride::Shape s2({1, 2, 6, 4});
        const upstride::Shape s3({1, 2});

        CHECK((!(s1 == s2)));
        CHECK((!(s1 == s3)));
        std::cout << std::endl;
    }
}

enum binop { plus,
             minus };
template <typename T>
bool accumulatorTest(const upstride::AllocatedTensor<upstride::device::CPU, T>& srcTensor, upstride::AllocatedTensor<upstride::device::CPU, T>& dstTensor, binop op) {
    upstride::AllocatedTensor<upstride::device::CPU, T> dstCopyTensor(dstTensor.getShape());

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

        upstride::AllocatedTensor<upstride::device::CPU, float> t1(s1);

        float* t1Ptr = t1.getDataPtr();
        // Ensure values are not zero
        for (int i = 0; i < numel; i++)
            t1Ptr[i] = i * 3.0f;

        // Verify if values were indeed modified before applying t1.zero()
        bool test = true;
        for (int i = 0; i < numel && test; i++) {
            if (t1.getDataPtr()[i] != i * 3.0f)
                test = false;
        }

        t1.zero();
        // Verify if new values are zero
        for (int i = 0; i < numel && test; i++) {
            if (t1.getDataPtr()[i] != 0.0f)
                test = false;
        }
        CHECK((test));
        std::cout << std::endl;
    }

    SUBCASE(" Test: [float/int] Tensor Src + Dst") {
        std::cout << "[float] Test: Tensor Src + Dst" << std::endl;
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(s1), dstTensori(s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(s1), dstTensorf(s1);
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
        upstride::AllocatedTensor<upstride::device::CPU, float> srcTensori(s1), dstTensori(s1);
        upstride::AllocatedTensor<upstride::device::CPU, int> srcTensorf(s1), dstTensorf(s1);
        setRandVal(srcTensori);
        setRandVal(dstTensori);
        setRandVal(srcTensorf);
        setRandVal(dstTensorf);

        CHECK((accumulatorTest(srcTensori, dstTensori, binop::minus)));
        CHECK((accumulatorTest(srcTensorf, dstTensorf, binop::minus)));
        std::cout << std::endl;
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

        const int typeDim = 4;
        const upstride::DataFormat df = upstride::DataFormat::NCHW;

        const upstride::Shape inputShape({1, 224, 224, 3});
        const upstride::Shape kernelShape({4, 3, 3, 3, 32});
        const upstride::Shape expectedShape({1, 3, 224, 3});

        upstride::Padding paddingPreset = upstride::Padding::SAME;
        const std::vector<int32_t>& explicitPadding = {0, 0};
        const std::vector<int32_t>& stride = {1, 1};
        const std::vector<int32_t>& dilation = {0, 0};
        upstride::IntPair padBefore, padAfter;

        upstride::Shape outputShape = upstride::computeConvOutputSize(typeDim,
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

    upstride::Tensor<upstride::device::CPU, const int> testInputTensor(TEST_DATA_SHAPE, TEST_DATA);

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
