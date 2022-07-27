#include <float.h>
#include <stdio.h>
#include <unistd.h>

#include "convolution_param.h"

#define IS_FLOAT (param.mType == 0)

template <class T, class T2>
__global__ void ConvKernelNaive(
    T* dst, const T* src, ConvolutionParam param, int output_h, int output_w,
    T* kernel, T* bias) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int input_channel = param.mInputChannel;
    int output_channel = param.mOutputChannel;
    int group = param.mGroup;
    int h = param.mH;
    int w = param.mW;
    int kernel_h = param.mKernelH;
    int kernel_w = param.mKernelW;
    int stride_h = param.mStrideH;
    int stride_w = param.mStrideW;
    int padding_h = param.mPaddingH;
    int padding_w = param.mPaddingW;
    int dilation_h = param.mDilationH;
    int dilation_w = param.mDilationW;

    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    if (channel >= output_channel || output_x >= output_h ||
        output_y >= output_w) {
        return;
    }
    // input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // NCHW
    T2 sum = (T2)0;
    if (bias != NULL) {
        sum += bias[channel];
    }
    if (group == 1) {
        for (int k = 0; k < input_channel; k++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int orig_x = output_x * stride_h + i * dilation_h;
                    int orig_y = output_y * stride_w + j * dilation_w;

                    T src_value = (T)0;
                    if (orig_x >= padding_h && orig_x < padding_h + h &&
                        orig_y >= padding_w && orig_y < padding_w + w) {
                        src_value =
                            src[k * h * w + (orig_x - padding_h) * w + orig_y -
                                padding_w];
                    }
                    // OIHW
                    T kernel_value = (T)0;
                    kernel_value = kernel
                        [channel * input_channel * kernel_h * kernel_w +
                         k * kernel_h * kernel_w + i * kernel_w + j];
                    sum += src_value * kernel_value;
                }
            }
        }
    } else {
        int step = input_channel / output_channel;
        for (int k = channel * step; k < channel * step + step; k++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int orig_x = output_x * stride_h + i * dilation_h;
                    int orig_y = output_y * stride_w + j * dilation_w;

                    T src_value = (T)0;
                    if (orig_x >= padding_h && orig_x < padding_h + h &&
                        orig_y >= padding_w && orig_y < padding_w + w) {
                        src_value =
                            src[k * h * w + (orig_x - padding_h) * w + orig_y -
                                padding_w];
                    }
                    // OIHW
                    T kernel_value =
                        kernel[k * kernel_h * kernel_w + i * kernel_w + j];
                    sum += src_value * kernel_value;
                }
            }
        }
    }
    if (!IS_FLOAT) {
        sum = (T)(
            sum * param.mInputScale * param.mKernelScale / param.mOutputScale);
    }
    dst[channel * output_h * output_w + output_x * output_w + output_y] = sum;
}

void ConvolutionNaive(
    void* dst, const void* src, struct ConvolutionParam param, void* kernel,
    void* bias, void* workspace, cudaStream_t stream) {
    //  input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // 20, 24, 24
    int input_channel = param.mInputChannel;
    int output_channel = param.mOutputChannel;
    int group = param.mGroup;
    int h = param.mH;
    int w = param.mW;
    int kernel_h = param.mKernelH;
    int kernel_w = param.mKernelW;
    int stride_h = param.mStrideH;
    int stride_w = param.mStrideW;
    int padding_h = param.mPaddingH;
    int padding_w = param.mPaddingW;
    int dilation_h = param.mDilationH;
    int dilation_w = param.mDilationW;

    // NOTE: `floor` for convolution
    int output_h =
        (h - (dilation_h * (kernel_h - 1) + 1) + 2 * padding_h) / stride_h + 1;
    int output_w =
        (w - (dilation_w * (kernel_w - 1) + 1) + 2 * padding_w) / stride_w + 1;

    int total_size = output_channel * output_h * output_w;
    if (IS_FLOAT) {
        ConvKernelNaive<float, float>
            <<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
                (float*)dst, (const float*)src, param, output_h, output_w,
                (float*)kernel, (float*)bias);
    } else {
        ConvKernelNaive<int8_t, int>
            <<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
                (int8_t*)dst, (const int8_t*)src, param, output_h, output_w,
                (int8_t*)kernel, (int8_t*)bias);
    }
}
