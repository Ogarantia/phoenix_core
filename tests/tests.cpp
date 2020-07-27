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
// #include <random>

#include "doctest/doctest.h"
#include "upstride.hpp"

/* =============================================================================
                                 PHOENIX 
============================================================================= */

TEST_CASE("Test:Shape") {
    std::cout << "---- Test: Shape creation" << std::endl;

    SUBCASE(" Test: Shape::Shape()") {
        std::cout << " Test: Shape::Shape()" << std::endl;
        const int shapes[4] = {1, 2, 3, 4};
        upstride::Shape s1(4, shapes);

        std::cout << s1 << std::endl;

        CHECK((s1.getShapePtr()[0] == 1));
        CHECK((s1.getShapePtr()[1] == 2));
        CHECK((s1.getShapePtr()[2] == 3));
        CHECK((s1.getShapePtr()[3] == 4));
        std::cout << std::endl;
    }

    SUBCASE(" Test: Shape == Shape") {
        std::cout << " Test: Shape::operator==" << std::endl;
        const int shapes[4] = {1, 2, 3, 4};
        const upstride::Shape s1(4, shapes);
        const upstride::Shape s2(4, shapes);

        CHECK((s1 == s2));
        std::cout << std::endl;
    }

    SUBCASE(" Test: Shape !(==) Shape") {
        std::cout << " Test: !(Shape::operator==)" << std::endl;
        const int shape[4] = {1, 2, 3, 4};
        const int shape2[4] = {1, 2, 6, 4};
        const int shape3[2] = {1, 2};
        const upstride::Shape s1(4, shape);
        const upstride::Shape s2(4, shape2);
        const upstride::Shape s3(2, shape3);

        CHECK((!(s1 == s2)));
        CHECK((!(s1 == s3)));
        std::cout << std::endl;
    }
}

TEST_CASE("Test:Tensor") {
    std::cout << "---- Test: Tensor creation" << std::endl;

    SUBCASE(" Test: Tensor::Tensor()") {
        std::cout << " Test: Tensor::Tensor()" << std::endl;

        const int shapes[4] = {1, 2, 3, 4};
        upstride::Shape s1(4, shapes);
        float* img = (float*)calloc(224 * 224 * 3, sizeof(float));

        upstride::Tensor<float> t1(s1, img);
        bool test = true;

        for (int i = 0; i < 224 * 224 * 3 && test; i++) {
            if (t1.getDataPtr()[i] != 0.0f) {
                test = false;
            }
        }

        CHECK(( test ));
        CHECK(( t1.getShape() == s1 ));
        std::cout << std::endl;
    }
}

#if 1
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
#endif
#if 1
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

#endif
#if 1
TEST_CASE("Test:Utils") {
    std::cout << "---- Test: Utils functions" << std::endl;

    SUBCASE(" Test: Utils::computeConvOutputSize") {
        std::cout << " Test: Utils::computeConvOutputSize" << std::endl;

        const int typeDim = 4;
        const upstride::DataFormat df = upstride::DataFormat::NCHW;

        const int shapesI[4] = {1, 224, 224, 3};
        const upstride::Shape inputShape(4, shapesI);

        const int shapesK[5] = {4, 3, 3, 3, 32};
        const upstride::Shape kernelShape(5, shapesK);

        const int shapesE[4] = {1, 3, 224, 3};
        const upstride::Shape ExpectedShape(4, shapesE);

        upstride::Padding paddingPreset = upstride::Padding::SAME;
        const std::vector<int32_t>& explicitPadding = {0, 0};
        const std::vector<int32_t>& stride = {1, 1};
        const std::vector<int32_t>& dilation = {0, 0};

        upstride::Shape outputShape = upstride::computeConvOutputSize(typeDim,
                                                                      df,
                                                                      inputShape,
                                                                      kernelShape,
                                                                      paddingPreset,
                                                                      explicitPadding,
                                                                      stride,
                                                                      dilation);

        std::cout << outputShape << std::endl;

        CHECK((outputShape == ExpectedShape));
        std::cout << std::endl;
    }
}

#endif
#if 0
TEST_CASE("Test:Conv2D") {
    std::cout << "---- Test: Conv2D OneDNN version" << std::endl;

    SUBCASE(" Test: upstride::UpstrideConv2DFunctor<device::CPU, float>") {
        std::cout << " Test: upstride::UpstrideConv2DFunctor<device::CPU, float>" << std::endl;

        float* input = (float*)calloc(224 * 224 * 3, sizeof(float));
        float* kernel = (float*)calloc(3 * 3 * 32, sizeof(float));

        // int outputSize = upstride::computeConvOutputSize(
        //         1, dataFormat,
        //         input.getShape(), filter.getShape(),
        //         paddingPreset, explicitPadding, stride, dilation));
        // float* output = (float*)calloc(3 * 3 * 32, sizeof(float));

        upstride::UpstrideConv2DFunctor<upstride::device::CPU, float> conv2d();
        // upstride_ops.upstride_conv2d(
        //     tf.zeros((1, 224, 224, 3), dtype = tf.float32),
        //     tf.zeros((1, 3, 3, 3, 32), dtype = tf.float32),
        //     strides = [ 1, 1 ],
        //     padding = 'SAME')

        std::cout << " COUCOU   OneDNN    : " << std::endl;
        std::cout << std::endl;

        // CHECK((expectedOut1 == a + d));
        std::cout << std::endl;
    }
}
#endif