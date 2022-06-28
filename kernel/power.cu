#include <stdio.h>

__global__ void PowerKernel(
    float* output, const float* input, float scale, float power, float shift,
    int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        output[id] = pow(shift + scale * input[id], power);
    }
}

void Power(
    float* output, const float* input, float scale, float power, float shift,
    int N, cudaStream_t stream) {
    int block = int(N / 128) + 1;
    PowerKernel<<<block, 128, 0, stream>>>(
        output, input, scale, power, shift, N);
}
