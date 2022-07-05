#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "batch_norm_param.h"

__global__ void BatchNormKernel(
    float* dst, const float* src, BatchNormParam param, float* mean,
    float* var) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int h = param.mH;
    int w = param.mW;
    int channel = param.mChannel;
    float eps = param.mEps;
    float average = param.mMovingAverage;

    if (global_id >= channel * h * w) {
        return;
    }
    int output_c = global_id / h / w;
    int output_x = global_id % (h * w) / w;
    int output_y = global_id % (h * w) % w;

    dst[output_c * h * w + output_x * w + output_y] =
        (src[output_c * h * w + output_x * w + output_y] -
         mean[output_c] / average) /
        sqrt(var[output_c] / average + eps);
}

void BatchNorm(
    float* dst, const float* src, BatchNormParam param, float* mean, float* var,
    void* workspace, cudaStream_t stream) {
    int h = param.mH;
    int w = param.mW;
    int channel = param.mChannel;

    float* meanWeights = (float*)workspace;
    float* varWeights = (float*)workspace + channel;
    cudaMemcpy(meanWeights, mean, channel * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(varWeights, var, channel * 4, cudaMemcpyHostToDevice);

    int total_size = channel * h * w;
    BatchNormKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, param, meanWeights, varWeights);
}
