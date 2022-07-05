#include <float.h>
#include <stdio.h>

__global__ void PermuteKernel(
    float* dst, const float* src, int total_size, int nb_dims, int* input_dims,
    int* input_mul, int* output_mul) {
    int input_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (input_id >= total_size) {
        return;
    }
    int output_id = 0;
    for (int i = 0; i < nb_dims; i++) {
        int x = (input_id / input_mul[i]) % input_dims[i];
        output_id += x * output_mul[i];
    }
    dst[output_id] = src[input_id];
}

void Permute(
    float* dst, const float* src, int nb_dims, int* input_dims, int* input_mul,
    int* output_mul, void* workspace, cudaStream_t stream) {
    int total_size = 1;
    for (int i = 0; i < nb_dims; i++) {
        total_size *= input_dims[i];
    }

    int* device_input_dims = (int*)workspace;
    int* device_input_mul = (int*)workspace + nb_dims;
    int* device_output_mul = (int*)workspace + 2 * nb_dims;

    cudaMemcpy(
        device_input_dims, input_dims, nb_dims * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_input_mul, input_mul, nb_dims * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_output_mul, output_mul, nb_dims * 4, cudaMemcpyHostToDevice);

    PermuteKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, total_size, nb_dims, device_input_dims, device_input_mul,
        device_output_mul);
}
