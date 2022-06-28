#include <assert.h>
#include <float.h>
#include <stdio.h>

__global__ void EltwiseKernel(
    float* dst, const float* src, const float* src2, int total_size,
    int operation) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= total_size) {
        return;
    }

    float tmp = 0.0;
    switch (operation) {
        case 0:  // prod
            tmp = src[global_id] * src2[global_id];
            break;
        case 1:  // sum
            tmp = src[global_id] + src2[global_id];
            break;            
        case 2:  // max
            tmp = max(src[global_id], src2[global_id]);
            break;
    }
    dst[global_id] = tmp;
}

void Eltwise(
    float* dst, const float* src, const float* src2, int total_size,
    int operation, cudaStream_t stream) {
    EltwiseKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, src2, total_size, operation);
}
