#include <float.h>
#include <stdio.h>

__global__ void Matmul(
    float* dst, const float* src, int input_size, int output_size,
    float* kernel, float* bias) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < output_size) {
        float sum = bias[id];
        for (int i = 0; i < input_size; i++) {
            sum += src[i] * kernel[id * input_size + i];
        }
        dst[id] = sum;
    }
}

void InnerProduct(
    float* dst, const float* src, int input_size, int output_size,
    float* kernel, float* bias, cudaStream_t stream) {
    float* kernelWeights;
    float* biasWeights;

    cudaMalloc(&kernelWeights, input_size * output_size * 4);
    cudaMalloc(&biasWeights, output_size * 4);
    cudaMemcpy(
        kernelWeights, kernel, input_size * output_size * 4,
        cudaMemcpyHostToDevice);
    cudaMemcpy(biasWeights, bias, output_size * 4, cudaMemcpyHostToDevice);

    // 800x500
    int threads = 128;
    int blocks = (int)(output_size / threads) + 1;
    Matmul<<<blocks, threads, 0, stream>>>(
        dst, src, input_size, output_size, kernelWeights, biasWeights);
}
