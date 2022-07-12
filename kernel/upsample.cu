#include <stdio.h>

#include "upsample_param.h"

__global__ void UpsampleKernel(
    float* output, const float* input, const float* mask, int total_size,
    int output_h, int output_w, struct UpsampleParam param) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    // mask:
    // output_blob(n,c,h,w) = 1, 64, 64, 128
    // 257.000000 258.000000 4.000000 7.000000 265.000000 267.000000
    // 269.000000 15.000000 17.000000 18.000000 21.000000 22.000000
    // 280.000000 26.000000 29.000000 30.000000 32511.000000 32764.000000
    // 32763.000000 32761.000000 32759.000000 32757.000000 32498.000000
    // 32497.000000 32494.000000 32493.000000 32490.000000 32744.000000
    // 32742.000000 32741.000000 32482.000000 32736.000000
    //
    // conv:
    // output_blob(n,c,h,w) = 1, 64, 64, 128
    // 0.006930 0.060615 0.131139 0.144623 0.338300 0.275927 0.463221 0.411572
    // 0.369539 0.412581 0.460310 0.487279 0.505763 0.487409 0.460202 0.421541
    // 0.217332 0.249646 0.225860 0.225513 0.191663 0.121349 0.159464 0.196097
    // 0.145067 0.185020 0.211315 0.188861 0.214563 0.239356 0.267270 0.308333
    //
    // upsample:
    // Processing time = 1314 ms
    // output_blob(n,c,h,w) = 1, 64, 128, 256
    // 0.000000 0.000000 0.000000 0.000000 0.131139 0.000000 0.000000 0.144623
    // 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.411572
    // 0.000000 0.000000 0.000000 0.249646 0.225860 0.000000 0.225513 0.000000
    // 0.191663 0.000000 0.121349 0.000000 0.000000 0.000000 0.000000 0.000000

    if (global_id >= total_size) {
        return;
    }
    int orig_x = output_x / param.mScale;
    int orig_y = output_y / param.mScale;
    int orig_offset =
        channel * param.mH * param.mW + orig_x * param.mW + orig_y;

    int output_offset_per_channel = output_x * output_w + output_y;

    int mask_index = (int)(mask[orig_offset]);
    float mask_value = input[orig_offset];

    output[global_id] = 0.0f;
    if (output_offset_per_channel == mask_index) {
        output[global_id] = mask_value;
    }
}

void Upsample(
    float* dst, const float* src, const float* mask, struct UpsampleParam param,
    void* workspace, cudaStream_t stream) {
    int output_h = param.mH * param.mScale;
    int output_w = param.mW * param.mScale;
    int total_size = param.mChannel * output_h * output_w;
    UpsampleKernel<<<int(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, mask, total_size, output_h, output_w, param);
}
