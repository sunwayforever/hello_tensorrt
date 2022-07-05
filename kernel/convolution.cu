#include <float.h>
#include <stdio.h>
#include <unistd.h>

__global__ void ConvKernel(
    float* dst, const float* src, int input_channel, int output_channel,
    int group, int h, int w, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int output_h, int output_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, float* kernel, float* bias) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    if (channel >= output_channel || output_x >= output_h ||
        output_y >= output_w) {
        return;
    }
    // input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // NCHW
    if (group == 1) {
        float sum = 0.0f;
        if (bias != NULL) {
            sum += bias[channel];
        }
        for (int k = 0; k < input_channel; k++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int orig_x = output_x * stride_h + i * dilation_h;
                    int orig_y = output_y * stride_w + j * dilation_w;

                    float src_value = 0.0;
                    if (orig_x >= padding_h && orig_x < padding_h + h &&
                        orig_y >= padding_w && orig_y < padding_w + w) {
                        src_value =
                            src[k * h * w + (orig_x - padding_h) * w + orig_y -
                                padding_w];
                    }
                    // OIHW
                    float kernel_value = kernel
                        [channel * input_channel * kernel_h * kernel_w +
                         k * kernel_h * kernel_w + i * kernel_w + j];
                    sum += src_value * kernel_value;
                }
            }
        }
        dst[channel * output_h * output_w + output_x * output_w + output_y] =
            sum;
    } else {
        int step = input_channel / output_channel;
        float sum = 0.0f;
        if (bias != NULL) {
            sum += bias[channel];
        }
        for (int k = channel * step; k < channel * step + step; k++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int orig_x = output_x * stride_h + i * dilation_h;
                    int orig_y = output_y * stride_w + j * dilation_w;

                    float src_value = 0.0;
                    if (orig_x >= padding_h && orig_x < padding_h + h &&
                        orig_y >= padding_w && orig_y < padding_w + w) {
                        src_value =
                            src[k * h * w + (orig_x - padding_h) * w + orig_y -
                                padding_w];
                    }
                    // OIHW
                    float kernel_value =
                        kernel[k * kernel_h * kernel_w + i * kernel_w + j];
                    sum += src_value * kernel_value;
                }
            }
        }
        dst[channel * output_h * output_w + output_x * output_w + output_y] =
            sum;
    }
}

void Convolution(
    float* dst, const float* src, int input_channel, int output_channel,
    int group, int h, int w, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,
    float* kernel, float* bias, void* workspace, cudaStream_t stream) {
    float* kernelWeights = (float*)workspace;
    float* biasWeights = NULL;
    //  input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // 20, 24, 24
    if (group == 1) {
        cudaMemcpy(
            kernelWeights, kernel,
            input_channel * output_channel * kernel_h * kernel_w * 4,
            cudaMemcpyHostToDevice);
        workspace = (float*)workspace +
                    input_channel * output_channel * kernel_h * kernel_w;
    } else {
        cudaMemcpy(
            kernelWeights, kernel, input_channel * kernel_h * kernel_w * 4,
            cudaMemcpyHostToDevice);
        workspace = (float*)workspace + input_channel * kernel_h * kernel_w;
    }
    if (bias != NULL) {
        biasWeights = (float*)workspace;
        cudaMemcpy(
            biasWeights, bias, output_channel * 4, cudaMemcpyHostToDevice);
    }

    // NOTE: `floor` for convolution
    int output_h =
        (h - (dilation_h * (kernel_h - 1) + 1) + 2 * padding_h) / stride_h + 1;
    int output_w =
        (w - (dilation_w * (kernel_w - 1) + 1) + 2 * padding_w) / stride_w + 1;

    int total_size = output_channel * output_h * output_w;
    ConvKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, input_channel, output_channel, group, h, w, kernel_h,
        kernel_w, stride_h, stride_w, output_h, output_w, padding_h, padding_w,
        dilation_h, dilation_w, kernelWeights, biasWeights);
}
