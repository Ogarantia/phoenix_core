#include "tensor.hpp"

namespace upstride {

void TensorManipulations<device::CPU>::accumulateAdd(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output) {
    int shapeNumel = input.getShape().numel();
    float* outputPtr = output.getDataPtr();
    const float* inputPtr = input.getDataPtr();
    int idx = 0;
#if __AVX512F__
    for (; idx < shapeNumel - 15; idx += 16) {
        __m512 src_v = _mm512_loadu_ps(&inputPtr[idx]);
        __m512 dst_v = _mm512_loadu_ps(&outputPtr[idx]);
        dst_v = _mm512_add_ps(dst_v, src_v);
        _mm512_storeu_ps(&outputPtr[idx], dst_v);
    }
#elif __AVX__
    for (; idx < shapeNumel - 7; idx += 8) {
        __m256 src_v = _mm256_loadu_ps(&inputPtr[idx]);
        __m256 dst_v = _mm256_loadu_ps(&outputPtr[idx]);
        dst_v = _mm256_add_ps(dst_v, src_v);
        _mm256_storeu_ps(&outputPtr[idx], dst_v);
    }
#elif __SSE4_1__
    for (; idx < shapeNumel - 3; idx += 4) {
        __m128 src_v = _mm_loadu_ps(&inputPtr[idx]);
        __m128 dst_v = _mm_loadu_ps(&outputPtr[idx]);
        dst_v = _mm_add_ps(dst_v, src_v);
        _mm_storeu_ps(&outputPtr[idx], dst_v);
    }
#endif
    for (; idx < shapeNumel; idx++) {
        outputPtr[idx] += inputPtr[idx];
    }
}

void TensorManipulations<device::CPU>::accumulateAdd(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output) {
    int shapeNumel = input.getShape().numel();
    int* outputPtr = output.getDataPtr();
    const int* inputPtr = input.getDataPtr();
    int idx = 0;
#if __AVX512F__
    for (; idx < shapeNumel - 15; idx += 16) {
        __m512i src_v = _mm512_loadu_epi32(&inputPtr[idx]);
        __m512i dst_v = _mm512_loadu_epi32(&outputPtr[idx]);
        dst_v = _mm512_add_epi32(dst_v, src_v);
        _mm512_storeu_epi32(&outputPtr[idx], dst_v);
    }
#elif __AVX2__
    for (; idx < shapeNumel - 7; idx += 8) {
        __m256i src_v = _mm256_loadu_si256((const __m256i*)&inputPtr[idx]);
        __m256i dst_v = _mm256_loadu_si256((const __m256i*)&outputPtr[idx]);
        dst_v = _mm256_add_epi32(dst_v, src_v);
        _mm256_storeu_si256((__m256i*)&outputPtr[idx], dst_v);
    }
#elif __SSE4_1__
    for (; idx < shapeNumel - 3; idx += 4) {
        __m128i src_v = _mm_loadu_si128((const __m128i*)&inputPtr[idx]);
        __m128i dst_v = _mm_loadu_si128((const __m128i*)&outputPtr[idx]);
        dst_v = _mm_add_epi32(dst_v, src_v);
        _mm_storeu_si128((__m128i*)&outputPtr[idx], dst_v);
    }
#endif
    for (; idx < shapeNumel; idx++) {
        outputPtr[idx] += inputPtr[idx];
    }
}

void TensorManipulations<device::CPU>::accumulateSub(const Tensor<device::CPU, float>& input, Tensor<device::CPU, float>& output) {
    int shapeNumel = input.getShape().numel();
    float* outputPtr = output.getDataPtr();
    const float* inputPtr = input.getDataPtr();
    int idx = 0;
#if __AVX512F__
    for (; idx < shapeNumel - 15; idx += 16) {
        __m512 src_v = _mm512_loadu_ps(&inputPtr[idx]);
        __m512 dst_v = _mm512_loadu_ps(&outputPtr[idx]);
        dst_v = _mm512_sub_ps(dst_v, src_v);
        _mm512_storeu_ps(&outputPtr[idx], dst_v);
    }
#elif __AVX__
    for (; idx < shapeNumel - 7; idx += 8) {
        __m256 src_v = _mm256_loadu_ps(&inputPtr[idx]);
        __m256 dst_v = _mm256_loadu_ps(&outputPtr[idx]);
        dst_v = _mm256_sub_ps(dst_v, src_v);
        _mm256_storeu_ps(&outputPtr[idx], dst_v);
    }
#elif __SSE4_1__
    for (; idx < shapeNumel - 3; idx += 4) {
        __m128 src_v = _mm_loadu_ps(&inputPtr[idx]);
        __m128 dst_v = _mm_loadu_ps(&outputPtr[idx]);
        dst_v = _mm_sub_ps(dst_v, src_v);
        _mm_storeu_ps(&outputPtr[idx], dst_v);
    }
#endif
    for (; idx < shapeNumel; idx++) {
        outputPtr[idx] -= inputPtr[idx];
    }
}

void TensorManipulations<device::CPU>::accumulateSub(const Tensor<device::CPU, int>& input, Tensor<device::CPU, int>& output) {
    int shapeNumel = input.getShape().numel();
    int* outputPtr = output.getDataPtr();
    const int* inputPtr = input.getDataPtr();
    int idx = 0;
#if __AVX512F__
    for (; idx < shapeNumel - 15; idx += 16) {
        __m512i src_v = _mm512_loadu_epi32(&inputPtr[idx]);
        __m512i dst_v = _mm512_loadu_epi32(&outputPtr[idx]);
        dst_v = _mm512_sub_epi32(dst_v, src_v);
        _mm512_storeu_epi32(&outputPtr[idx], dst_v);
    }
#elif __AVX2__
    for (; idx < shapeNumel - 7; idx += 8) {
        __m256i src_v = _mm256_loadu_si256((const __m256i*)&inputPtr[idx]);
        __m256i dst_v = _mm256_loadu_si256((const __m256i*)&outputPtr[idx]);
        dst_v = _mm256_sub_epi32(dst_v, src_v);
        _mm256_storeu_si256((__m256i*)&outputPtr[idx], dst_v);
    }
#elif __SSE4_1__
    for (; idx < shapeNumel - 3; idx += 4) {
        __m128i src_v = _mm_loadu_si128((const __m128i*)&inputPtr[idx]);
        __m128i dst_v = _mm_loadu_si128((const __m128i*)&outputPtr[idx]);
        dst_v = _mm_sub_epi32(dst_v, src_v);
        _mm256_storeu_si256((__m256i*)&outputPtr[idx], dst_v);
    }
#endif
    for (; idx < shapeNumel; idx++) {
        outputPtr[idx] -= inputPtr[idx];
    }
}

}  // namespace upstride