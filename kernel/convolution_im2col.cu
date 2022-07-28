#include <cublas_v2.h>
#include <float.h>
#include <stdio.h>
#include <unistd.h>

#include "convolution_param.h"

#define IS_FLOAT (param.mType == 0)

__global__ void Im2ColKernel(
    float* dst, const float* src, ConvolutionParam param, int output_h,
    int output_w) {
    // NOTE:
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int global_id = blockIdx.x * blockDim.x + threadIdx.x;
         global_id < output_h * output_w; global_id += gridDim.x * blockDim.x) {
        int input_channel = param.mInputChannel;
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

        int output_x = global_id / output_w;
        int output_y = global_id % output_w;

        int index = 0;
        for (int k = 0; k < input_channel; k++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int orig_x = output_x * stride_h + i * dilation_h;
                    int orig_y = output_y * stride_w + j * dilation_w;

                    float src_value = (float)0;
                    if (orig_x >= padding_h && orig_x < padding_h + h &&
                        orig_y >= padding_w && orig_y < padding_w + w) {
                        src_value =
                            src[k * h * w + (orig_x - padding_h) * w + orig_y -
                                padding_w];
                    }
                    dst[index * (output_h * output_w) + output_x * output_w +
                        output_y] = src_value;
                    index += 1;
                }
            }
        }
    }
}

__global__ void BroadcastBiasKernel(
    float* dst, float* bias, ConvolutionParam param, int output_h,
    int output_w) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int output_channel = param.mOutputChannel;

    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    if (channel >= output_channel || output_x >= output_h ||
        output_y >= output_w) {
        return;
    }

    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        bias[channel];
}

void ConvolutionIm2Col(
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

    float* im2col_dst;
    cudaMalloc(
        &im2col_dst,
        output_h * output_w * kernel_h * kernel_w * input_channel * 4);

    Im2ColKernel<<<10, 1024, 0, stream>>>(
        im2col_dst, (const float*)src, param, output_h, output_w);

    if (bias != NULL) {
        BroadcastBiasKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
            (float*)dst, (float*)bias, param, output_h, output_w);
    }
    // im2col_dst: (X)hw
    // kernel: O(x)
    // bias: O
    // output: Ohw
    const float alf = 1.0f;
    const float bet = bias == NULL ? 0.0f : 1.0f;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);

    if (group == 1) {
        // kernel: [m,k], im2col: [k,n]  dst: [m,n]
        int m = output_channel;
        int n = output_h * output_w;
        int k = input_channel * kernel_h * kernel_w;

        // NOTE: cublas is `column-major`, so exchange A, B to get rid of
        // transposition. (because T(A)*T(B)=T(B*A)), also remember that now A
        // becomes [n,k], and B becomes [k,m]
        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, im2col_dst, n,
            (float*)kernel, k, beta, (float*)dst, n);
    } else {
        // kenrel: IHW -> (group)1(step*HW)
        // input: CHW -> (group)(step*HW)hw
        int step = input_channel / output_channel;
        int m = 1;
        int n = output_h * output_w;
        int k = step * kernel_h * kernel_w;
        // NOTE: this is a naive implementation of `batched matmul`, cublas
        // should have a faster one
        for (int i = 0; i < group; i++) {
            cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha,
                im2col_dst + i * n * k, n, (float*)kernel + i * k, k, beta,
                ((float*)dst) + i * n, n);
        }
    }

    cublasDestroy(handle);
}
