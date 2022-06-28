#include <stdio.h>

__global__ void ReluKernel(float* output, float* input, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (input[id] > 0.0) {
            output[id] = input[id];
        } else {
            output[id] = 0.0;
        }
    }
}

void Relu(float* output, float* input, int N, cudaStream_t stream) {
    int block = int(N / 128) + 1;
    ReluKernel<<<block, 128, 0, stream>>>(output, input, N);
}
