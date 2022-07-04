#include <float.h>
#include <stdio.h>
#include <thrust/extrema.h>

__global__ void ConvKernel(
    int8_t* dst, const int8_t* src, int input_scale, int output_scale,
    int input_channel, int output_channel, int h, int w, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int output_h, int output_w,
    int padding_h, int padding_w, int8_t* kernel, int kernel_scale, float* bias) {
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
    float sum = 0.0f;
    if (bias != NULL) {
        sum += bias[channel];
    }
    for (int k = 0; k < input_channel; k++) {
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                int orig_x = output_x * stride_h + i;
                int orig_y = output_y * stride_w + j;

                int8_t src_value = 0;
                if (orig_x >= padding_h && orig_x < padding_h + h &&
                    orig_y >= padding_w && orig_y < padding_w + w) {
                    src_value =
                        src[k * h * w + (orig_x - padding_h) * w + orig_y -
                            padding_w];
                }
                // OIHW
                int8_t kernel_value = kernel
                    [channel * input_channel * kernel_h * kernel_w +
                     k * kernel_h * kernel_w + i * kernel_w + j];
                sum += (float)src_value * kernel_value;
            }
        }
    }

    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        (int)(sum / input_scale / kernel_scale * output_scale );
}

void ConvolutionInt8(
    int8_t* dst, const int8_t* src, int input_scale, int output_scale,
    int input_channel, int output_channel, int group, int h, int w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h,
    int padding_w, float* kernel, float* bias, cudaStream_t stream) {
    int8_t* kernelWeights;
    float* biasWeights;
    //  input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // 20, 24, 24
    float kernel_max = kernel[0];
    float kernel_min = kernel[0];
    int kernel_scale = 0;
    int8_t* kernel_I8 = malloc(sizeof(int8_t)*(input_channel * output_channel * kernel_h * kernel_w));
    for(int i=0, i<input_channel * output_channel * kernel_h * kernel_w, i++){
        if (kernel[i] > kernel_max){
            kernel_max = kernel[i];
        }
        if (kernel[i] < kernel_min){
            kernel_min = kernel[i];
        }
    }

    kernel_scale = max(abs(kernel_max),abs(kernel_min))/127;
    
    for (int i=0, i<input_channel * output_channel * kernel_h * kernel_w, i++){
        kernel_I8[i] = (int8_t)(kernel[i]/kernel_scale);
    }

    cudaMalloc(
        &kernelWeights,
        input_channel * output_channel * kernel_h * kernel_w * sizeof(int8_t));
    cudaMalloc(&biasWeights, output_channel * 4);
    cudaMemcpy(
        kernelWeights, kernel_I8,
        input_channel * output_channel * kernel_h * kernel_w  * sizeof(int8_t),
        cudaMemcpyHostToDevice);
    cudaMemcpy(biasWeights, bias, output_channel * 4, cudaMemcpyHostToDevice);

    // NOTE: `floor` for convolution
    int output_h = (h - kernel_h + 2 * padding_h) / stride_h + 1;
    int output_w = (w - kernel_w + 2 * padding_w) / stride_w + 1;

    int total_size = output_channel * output_h * output_w;
    ConvKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, input_scale, output_scale, input_channel, output_channel, h,
        w, kernel_h, kernel_w, stride_h, stride_w, output_h, output_w,
        padding_h, padding_w, kernelWeights, kernel_scale, biasWeights);
}
