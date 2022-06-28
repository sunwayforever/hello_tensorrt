#include <assert.h>
#include <float.h>
#include <stdio.h>

__global__ void BatchNormKernel(
    float* dst, const float* src, int channel, int h, int w, float eps,
    float average, float* mean, float* var) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

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
    float* dst, const float* src, int channel, int h, int w, float eps,
    float average, float* mean, float* var, cudaStream_t stream) {
    float* meanWeights;
    float* varWeights;

    cudaMalloc(&meanWeights, channel * 4);
    cudaMalloc(&varWeights, channel * 4);
    cudaMemcpy(meanWeights, mean, channel * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(varWeights, var, channel * 4, cudaMemcpyHostToDevice);

    int total_size = channel * h * w;
    BatchNormKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, channel, h, w, eps, average, meanWeights, varWeights);
}
