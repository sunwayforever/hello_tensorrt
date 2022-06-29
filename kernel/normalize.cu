#include <assert.h>
#include <float.h>
#include <stdio.h>

__global__ void NormalizeKernel(
    float* dst, const float* src, float* sum, int channel, int h, int w,
    int across_spatial, int channel_shared, float eps, float* scale) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    assert(!across_spatial);
    assert(!channel_shared);

    // NOTE: dst[c,h,w]=src[c,h,w]*scale[c] / sqrt(sum(h,w)+eps)
    // sum(h,w)=SUM_c=0^c=channel(pow(src[c,h,w],2))
    // 
    if (global_id >= channel * h * w) {
        return;
    }
    int output_channel = global_id / h / w;
    int output_x = global_id % (h * w) / w;
    int output_y = global_id % (h * w) % w;

    float orig = src[output_channel * h * w + output_x * w + output_y];
    dst[output_channel * h * w + output_x * w + output_y] =
        orig * scale[output_channel] / sqrt(sum[output_x * w + output_y] + eps);
}

__global__ void SumKernel(
    float* dst, const float* src, int channel, int h, int w) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    float sum = 0.0;
    for (int i = 0; i < channel; i++) {
        sum += src[i * h * w + x * w + y] * src[i * h * w + x * w + y];
    }
    dst[x * w + y] = sum;
}

void Normalize(
    float* dst, const float* src, int channel, int h, int w, int across_spatial,
    int channel_shared, float eps, float* scale, void* workspace,
    cudaStream_t stream) {
    int total_size = channel * h * w;

    float* scaleWeights = (float*)workspace;
    cudaMemcpy(scaleWeights, scale, channel * 4, cudaMemcpyHostToDevice);

    float* tmp = (float*)workspace + channel;

    SumKernel<<<dim3(h, w), 1, 0, stream>>>(tmp, src, channel, h, w);
    NormalizeKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, tmp, channel, h, w, across_spatial, channel_shared, eps,
        scaleWeights);
}
