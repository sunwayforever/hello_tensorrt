#include <float.h>
#include <stdio.h>

#include <vector>
__global__ void PriorBoxKernel(
    float* output, int h, int w, float offset, float step, float* min_size,
    int min_size_count, float* max_size, int max_size_count,
    float* aspect_ratio, int aspect_ratio_count) {
    int output_x = blockIdx.x;
    int output_y = blockIdx.y;
    // #N of box is `aspect_ratio_count` + `max_size_count`
    int num_box = (aspect_ratio_count + 1) * min_size_count;
    float cx = (output_x + offset) * step;
    float cy = (output_y + offset) * step;

    int k = 0;
    // NOTE: output 的布局:
    // [ar=1, size=min_size[0]], [ar=1, size=min_size[1]], ...
    // [ar=1, size=sqrt(min_size[0]*max_size[0])], [ar=1,
    // size=sqrt(min_size[1]*max_size[1])], ... [ar=aspect_ratio[1],
    // size=min_size[0]], [ar=aspect_ratio[1], size=min_size[1]], ...
    // [ar=aspect_ratio[2], size=min_size[0]], [ar=aspect_ratio[1],
    // size=min_size[1]], ...
    // ...
    // [ar=aspect_ratio[N], ...
    int box_pos = (output_x * w + output_y) * num_box;
    int total_box_size = h * w * num_box * 4;
    for (int i = 0; i < min_size_count; i++) {
        // aspect_ratio=1, num_min[i]
        float box_size = min_size[i];
        float a = (cx - box_size / 2.0f) / 300.0f;
        float b = (cy - box_size / 2.0f) / 300.0f;
        float c = (cx + box_size / 2.0f) / 300.0f;
        float d = (cy + box_size / 2.0f) / 300.0f;
        output[(box_pos + k) * 4] = b;
        output[(box_pos + k) * 4 + 1] = a;
        output[(box_pos + k) * 4 + 2] = d;
        output[(box_pos + k) * 4 + 3] = c;
        // for variance
        output[total_box_size + (box_pos + k) * 4] = 0.1;
        output[total_box_size + (box_pos + k) * 4 + 1] = 0.1;
        output[total_box_size + (box_pos + k) * 4 + 2] = 0.2;
        output[total_box_size + (box_pos + k) * 4 + 3] = 0.2;
        k += 1;
    }
    for (int i = 0; i < min_size_count; i++) {
        // aspect_ratio=1, sqrt(min_size[i]*max_size[i])
        float box_size = sqrt(min_size[i] * max_size[i]);
        float a = (cx - box_size / 2.0f) / 300.0f;
        float b = (cy - box_size / 2.0f) / 300.0f;
        float c = (cx + box_size / 2.0f) / 300.0f;
        float d = (cy + box_size / 2.0f) / 300.0f;
        output[(box_pos + k) * 4] = b;
        output[(box_pos + k) * 4 + 1] = a;
        output[(box_pos + k) * 4 + 2] = d;
        output[(box_pos + k) * 4 + 3] = c;
        output[total_box_size + (box_pos + k) * 4] = 0.1;
        output[total_box_size + (box_pos + k) * 4 + 1] = 0.1;
        output[total_box_size + (box_pos + k) * 4 + 2] = 0.2;
        output[total_box_size + (box_pos + k) * 4 + 3] = 0.2;
        k += 1;
    }
    // for aspect ratios other than 1
    for (int i = 1; i < aspect_ratio_count; i++) {
        for (int j = 0; j < min_size_count; j++) {
            float box_size_h = min_size[j] / sqrt(aspect_ratio[i]);
            float box_size_w = min_size[j] * sqrt(aspect_ratio[i]);
            float a = (cx - box_size_h / 2.0f) / 300.0f;
            float b = (cy - box_size_w / 2.0f) / 300.0f;
            float c = (cx + box_size_h / 2.0f) / 300.0f;
            float d = (cy + box_size_w / 2.0f) / 300.0f;
            output[(box_pos + k) * 4] = b;
            output[(box_pos + k) * 4 + 1] = a;
            output[(box_pos + k) * 4 + 2] = d;
            output[(box_pos + k) * 4 + 3] = c;
            output[total_box_size + (box_pos + k) * 4] = 0.1;
            output[total_box_size + (box_pos + k) * 4 + 1] = 0.1;
            output[total_box_size + (box_pos + k) * 4 + 2] = 0.2;
            output[total_box_size + (box_pos + k) * 4 + 3] = 0.2;
            k += 1;
        }
    }
}

void PriorBox(
    float* dst, int h, int w, float offset, float step,
    std::vector<float> min_size, std::vector<float> max_size,
    std::vector<float> aspect_ratio, cudaStream_t stream) {
    //  h: 38 w: 38 mMinSize: 1:30  mMaxSize: 1:60  mAspectRatio: 3:1 2 0.5
    //  mStep: 8 mOffset: 0.5
    float* minSize;
    float* maxSize;
    float* aspectRatio;

    cudaMalloc(&minSize, min_size.size() * 4);
    cudaMalloc(&maxSize, max_size.size() * 4);
    cudaMalloc(&aspectRatio, aspect_ratio.size() * 4);

    cudaMemcpy(
        minSize, min_size.data(), min_size.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(
        maxSize, max_size.data(), max_size.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(
        aspectRatio, aspect_ratio.data(), aspect_ratio.size() * 4,
        cudaMemcpyHostToDevice);

    PriorBoxKernel<<<dim3(h, w), 1, 0, stream>>>(
        dst, h, w, offset, step, minSize, min_size.size(), maxSize,
        max_size.size(), aspectRatio, aspect_ratio.size());
}
