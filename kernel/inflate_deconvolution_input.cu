#include <float.h>
#include <stdio.h>

#include "convolution_param.h"

__global__ void Copy(
    float* dst, const float* src, int total_size,
    struct ConvolutionParam param) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= total_size) {
        return;
    }

    int input_h = param.mH;
    int input_w = param.mW;
    int channel = global_id / input_h / input_w;
    int x = global_id % (input_h * input_w) / input_w;
    int y = global_id % (input_h * input_w) % input_w;

    int output_h = (param.mH - 1) * param.mStrideH + 1;
    int output_w = (param.mW - 1) * param.mStrideW + 1;
    int output_x = x * param.mStrideH;
    int output_y = y * param.mStrideW;

    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        src[global_id];
}

float* InflateDeconvolutionInput(
    const float* src, struct ConvolutionParam param, cudaStream_t stream) {
    float* dst;

    int total_size = param.mInputChannel * param.mH * param.mW;

    int output_h = (param.mH - 1) * param.mStrideH + 1;
    int output_w = (param.mW - 1) * param.mStrideW + 1;

    cudaMallocManaged(&dst, param.mInputChannel * output_h * output_w * 4);
    cudaMemset(dst, 0, param.mInputChannel * output_h * output_w * 4);
    Copy<<<int(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, total_size, param);

    return dst;
}
