#include <stdio.h>

__global__ void Exp(float* output, float* input, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        output[id] = exp(input[id]);
    }
}

__global__ void Divid(float* output, float* sum, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        output[id] /= *sum;
    }
}

__global__ void Sum(float* data, float* result, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        atomicAdd(result, data[id]);
    }
}

void Softmax(float* output, float* input, int N, cudaStream_t stream) {
    float* sum;
    cudaMalloc(&sum, sizeof(float));
    int block = int(N / 128) + 1;
    Exp<<<block, 128, 0, stream>>>(output, input, N);
    Sum<<<block, 128, 0, stream>>>(output, sum, N);
    Divid<<<block, 128, 0, stream>>>(output, sum, N);
    cudaFree(sum);
}
