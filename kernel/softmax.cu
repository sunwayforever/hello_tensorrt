#include <stdio.h>

__global__ void Exp(float* output, float* input, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        output[id] = exp(input[id]);
    }
}

__global__ void Divid(
    float* output, float* sum, int x, int y, int z, int x_mul, int y_mul,
    int z_mul) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= x * y * z) {
        return;
    }
    int output_x = (id / x_mul) % x;
    int output_y = (id / y_mul) % y;
    output[id] /= sum[output_x * y + output_y];
}

__global__ void Sum(
    float* output, float* result, int x, int y, int z, int x_mul, int y_mul,
    int z_mul) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= x * y) {
        return;
    }
    int output_x = id / y;
    int output_y = id % y;

    for (int i = 0; i < z; i++) {
        atomicAdd(
            &result[id],
            output[output_x * x_mul + output_y * y_mul + i * z_mul]);
    }
}

void Softmax(
    float* output, float* input, int* dims, int axis, cudaStream_t stream) {
    // total_sum=12764
    int x, y, z, x_mul, y_mul, z_mul;
    switch (axis) {
        case 0:
            // zxy
            z = dims[0];
            x = dims[1];
            y = dims[2];
            x_mul = y;
            y_mul = 1;
            z_mul = x * y;
            break;
        case 1:
            // xzy
            x = dims[0];
            z = dims[1];
            y = dims[2];
            x_mul = z * y;
            y_mul = 1;
            z_mul = y;
            break;
        case 2:
            // xyz
            x = dims[0];
            y = dims[1];
            z = dims[2];
            x_mul = z * y;
            y_mul = z;
            z_mul = 1;
            break;
    }
    int total_sum = x * y;
    float* sum;
    cudaMalloc(&sum, total_sum * 4);

    int N = x * y * z;
    Exp<<<int(N / 128) + 1, 128, 0, stream>>>(output, input, N);

    Sum<<<int(total_sum / 128) + 1, 128, 0, stream>>>(
        output, sum, x, y, z, x_mul, y_mul, z_mul);

    Divid<<<int(N / 128) + 1, 128, 0, stream>>>(
        output, sum, x, y, z, x_mul, y_mul, z_mul);
    cudaFree(sum);
}
