#include <assert.h>
#include <float.h>
#include <stdio.h>

__global__ void ScaleKernel(
    float* dst, const float* src, int channel, int h, int w, float* scale,
    float* bias) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= channel * h * w) {
        return;
    }
    int output_c = global_id / h / w;
    int output_x = global_id % (h * w) / w;
    int output_y = global_id % (h * w) % w;

    float bias_value = 0.0;
    if (bias != NULL) {
        bias_value = bias[output_c];
    }
    dst[output_c * h * w + output_x * w + output_y] =
        (src[output_c * h * w + output_x * w + output_y]) * scale[output_c] +
        bias_value;
}

void Scale(
    float* dst, const float* src, int channel, int h, int w, float* scale,
    float* bias, cudaStream_t stream) {
    float* scaleWeights;
    float* biasWeights;

    cudaMalloc(&scaleWeights, channel * 4);
    cudaMalloc(&biasWeights, channel * 4);
    cudaMemcpy(scaleWeights, scale, channel * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(biasWeights, bias, channel * 4, cudaMemcpyHostToDevice);

    int total_size = channel * h * w;
    ScaleKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, channel, h, w, scaleWeights, biasWeights);
}
