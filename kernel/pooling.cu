#include <float.h>
#include <stdio.h>

__global__ void Max(
    int total_size, float* dst, const float* src, int h, int w, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int output_h, int output_w,
    int padding_h, int padding_w) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= total_size) {
        return;
    }
    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    float max_value = -FLT_MAX;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            int orig_x = output_x * stride_h + i;
            int orig_y = output_y * stride_w + j;

            float curr_value = 0.0;
            if (orig_x >= padding_h && orig_x < padding_h + h &&
                orig_y >= padding_w && orig_y < padding_w + w) {
                curr_value =
                    src[channel * h * w + (orig_x - padding_h) * w + orig_y -
                        padding_w];
            }
            max_value = max(max_value, curr_value);
        }
    }
    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        max_value;
}

__global__ void Average(
    int total_size, float* dst, const float* src, int h, int w, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int output_h, int output_w,
    int padding_h, int padding_w) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= total_size) {
        return;
    }
    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    float sum_value = 0.0;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            int orig_x = output_x * stride_h + i;
            int orig_y = output_y * stride_w + j;

            float curr_value = 0.0;
            if (orig_x >= padding_h && orig_x < padding_h + h &&
                orig_y >= padding_w && orig_y < padding_w + w) {
                curr_value =
                    src[channel * h * w + (orig_x - padding_h) * w + orig_y -
                        padding_w];
            }
            sum_value += curr_value;
        }
    }
    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        sum_value / (kernel_h * kernel_w);
}

static int ceil(int a, int b) { return a / b + (a % b > 0); }

void Pooling(
    float* dst, const float* src, int channel, int h, int w, int method,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h,
    int padding_w, cudaStream_t stream) {
    int output_h = ceil(h - kernel_h + 2 * padding_h, stride_h) + 1;
    // int padding_bottom = (output_h - 1) * stride_h + kernel_h - h;
    int output_w = ceil(w - kernel_w + 2 * padding_w, stride_w) + 1;
    // int padding_right = (output_w - 1) * stride_w + kernel_w - w;

    int total_size = channel * output_h * output_w;

    if (method == 0) {
        Max<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
            total_size, dst, src, h, w, kernel_h, kernel_w, stride_h, stride_w,
            output_h, output_w, padding_h, padding_w);
    } else if (method == 1) {
        Average<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
            total_size, dst, src, h, w, kernel_h, kernel_w, stride_h, stride_w,
            output_h, output_w, padding_h, padding_w);
    }
}
