#include <assert.h>
#include <float.h>
#include <stdio.h>

__global__ void LRN(
    float* dst, const float* src, int channel, int h, int w, int local_size,
    float alpha, float beta) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= channel * h * w) {
        return;
    }
    //  channel: 64 h: 56 w: 56 local_size: 5 alpha: 0.0001 beta: 0.75
    int output_channel = global_id / h / w;
    int output_x = global_id % (h * w) / w;
    int output_y = global_id % (h * w) % w;

    // NOTE: https://oneapi-src.github.io/oneDNN/dev_guide_lrn.html
    int begin_channel = -(local_size - 1) / 2;
    int end_channel = (local_size + 1) / 2;

    float sum = 0.0;
    for (int i = begin_channel; i < end_channel; i++) {
        int c = output_channel + i;
        if (c < 0 || c >= channel) {
            continue;
        }
        sum += pow(src[c * h * w + output_x * w + output_y], 2);
    }
    float norm = pow(1 + alpha * sum / local_size, beta);
    dst[output_channel * h * w + output_x * w + output_y] =
        src[output_channel * h * w + output_x * w + output_y] / norm;
}

void LRN(
    float* dst, const float* src, int channel, int h, int w, int local_size,
    float alpha, float beta, cudaStream_t stream) {
    int total_size = channel * h * w;

    LRN<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, channel, h, w, local_size, alpha, beta);
}
