#include <stdio.h>

#include "prelu_param.h"

__global__ void PReLUKernel(
    float* output, const float* input, struct PReLUParam param, float* slope) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= param.mTotalSize) {
        return;
    }
    int channel = id / (param.mTotalSize / param.mChannel);
    float slope_value = 0.0;
    if (param.mChannelShared == 0) {
        slope_value = slope[channel];
    } else {
        slope_value = slope[0];
    }
    if (input[id] >= 0.0) {
        output[id] = input[id];
    } else {
        output[id] = input[id] * slope_value;
    }
}

void PReLU(
    float* dst, const float* src, struct PReLUParam param, float* slope_weights,
    void* workspace, cudaStream_t stream) {
    float* slope = (float*)workspace;
    cudaMemcpy(
        slope, slope_weights, param.mSlopeWeightsCount * 4,
        cudaMemcpyHostToDevice);
    PReLUKernel<<<int(param.mTotalSize / 128) + 1, 128, 0, stream>>>(
        dst, src, param, slope);
}
