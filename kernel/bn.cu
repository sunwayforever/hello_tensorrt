#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "bn_param.h"

__global__ void BNKernel(
    float* dst, const float* src, BNParam param, float* scale, float* shift) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int h = param.mH;
    int w = param.mW;
    int channel = param.mChannel;

    if (global_id >= channel * h * w) {
        return;
    }
    int output_c = global_id / h / w;
    int output_x = global_id % (h * w) / w;
    int output_y = global_id % (h * w) % w;

    dst[output_c * h * w + output_x * w + output_y] =
        (src[output_c * h * w + output_x * w + output_y] * scale[output_c]) +
        shift[output_c];
}

void BN(
    float* dst, const float* src, BNParam param, float* scale, float* shift,
    void* workspace, cudaStream_t stream) {
    int h = param.mH;
    int w = param.mW;
    int channel = param.mChannel;

    float* scaleWeights = (float*)workspace;
    float* shiftWeights = (float*)workspace + channel;
    cudaMemcpy(scaleWeights, scale, channel * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(shiftWeights, shift, channel * 4, cudaMemcpyHostToDevice);

    int total_size = channel * h * w;
    BNKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, param, scaleWeights, shiftWeights);
}
