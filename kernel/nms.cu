#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "nms_param.h"

__global__ void RemapKernel(
    float* output, const float* mbox_loc, const float* mbox_prior_box,
    int num_box) {
    // priorbox 的值为 (ymin,xmin,ymax,xmax)
    int box_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_id >= num_box) {
        return;
    }
    float prior_ymin = mbox_prior_box[box_id * 4];
    float prior_xmin = mbox_prior_box[box_id * 4 + 1];
    float prior_ymax = mbox_prior_box[box_id * 4 + 2];
    float prior_xmax = mbox_prior_box[box_id * 4 + 3];

    float prior_xcenter = (prior_xmin + prior_xmax) / 2.0f;
    float prior_ycenter = (prior_ymin + prior_ymax) / 2.0f;
    float prior_width = prior_ymax - prior_ymin;
    float prior_height = prior_xmax - prior_xmin;

    float loc_ymin = mbox_loc[box_id * 4];
    float loc_xmin = mbox_loc[box_id * 4 + 1];
    float loc_ymax = mbox_loc[box_id * 4 + 2];
    float loc_xmax = mbox_loc[box_id * 4 + 3];

    float box_xcenter = (loc_xmin)*0.1 * prior_height + prior_xcenter;
    float box_ycenter = (loc_ymin)*0.1 * prior_width + prior_ycenter;
    float box_height = exp(loc_xmax * 0.2) * prior_height;
    float box_width = exp(loc_ymax * 0.2) * prior_width;

    output[box_id * 6 + 2] = box_xcenter - box_height / 2.0;
    output[box_id * 6 + 3] = box_ycenter - box_width / 2.0;
    output[box_id * 6 + 4] = box_xcenter + box_height / 2.0;
    output[box_id * 6 + 5] = box_ycenter + box_width / 2.0;
}

__global__ void ArgmaxKernel(
    float* output, const float* mbox_conf, int num_box, float threshold) {
    int box_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_id >= num_box) {
        return;
    }
    float max_conf = threshold;
    int max_index = -1;
    for (int i = 1; i < 21; i++) {
        if (mbox_conf[box_id * 21 + i] >= max_conf) {
            max_conf = mbox_conf[box_id * 21 + i];
            max_index = i;
        }
    }
    output[box_id * 6] = (float)max_index;
    if (max_index == -1) {
        output[box_id * 6 + 1] = 0.0;
        output[box_id * 6 + 2] = 0.0;
        output[box_id * 6 + 3] = 0.0;
        output[box_id * 6 + 4] = 0.0;
        output[box_id * 6 + 5] = 0.0;
    } else {
        output[box_id * 6 + 1] = max_conf;
    }
}

__device__ float IOU(int a_index, int b_index, float* box_info) {
    float a_x1=box_info[a_index*6+2];
    float a_y1=box_info[a_index*6+3];
    float a_x2=box_info[a_index*6+4];
    float a_y2=box_info[a_index*6+5];

    float b_x1=box_info[b_index*6+2];
    float b_y1=box_info[b_index*6+3];
    float b_x2=box_info[b_index*6+4];
    float b_y2=box_info[b_index*6+5];

    float a_area=(a_x2-a_x1)*(a_y2-a_y1);
    float b_area=(b_x2-b_x1)*(b_y2-b_y1);

    float x1=max(a_x1, b_x1);
    float y1=max(a_y1,b_y1);
    float x2=min(a_x2,b_x2);
    float y2=min(a_y2,b_y2);
    float intersection=(x2-x1)*(y2-y1);

    return intersection/(a_area+b_area-intersection);
}

__global__ void NMSKernel(float* dst, float* box_info, int* sort_index, int top_k, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= top_k || sort_index[tid]==-1) {
        return;
    }
    int best_index=sort_index[0];
    if (tid==0) {
        dst[0]=0.0f;
        dst[1]=box_info[best_index*6];
        dst[2]=box_info[best_index*6+1];
        dst[3]=box_info[best_index*6+3];
        dst[4]=box_info[best_index*6+2];
        dst[5]=box_info[best_index*6+5];
        dst[6]=box_info[best_index*6+4];                                
        return;
    }    
    float iou=IOU(best_index, sort_index[tid], box_info);
    if (iou >= threshold) {
        sort_index[tid]=-1;
    }
}

__global__ void SortKernel(
    float* box_info, int* sort_index, int num_box, int is_odd) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > num_box / 2) {
        return;
    }

    int a_index = 2 * tid + is_odd;
    int b_index = a_index + 1;
    if (b_index >= num_box) {
        return;
    }
    if (box_info[sort_index[a_index] * 6 + 1] <
        box_info[sort_index[b_index] * 6 + 1]) {
        int tmp = sort_index[a_index];
        sort_index[a_index] = sort_index[b_index];
        sort_index[b_index] = tmp;
    }
}

void NMS(
    float* dst, const float* mbox_loc, const float* mbox_conf,
    const float* mbox_priorbox, struct NMSParam param, void* workspace,
    cudaStream_t stream) {
    int num_box = param.mNumBox;

    // output: [class, conf, xmin, ymin, xmax, ymax]
    float* device_box_info;
    cudaMalloc(&device_box_info, num_box * 6 * 4);

    RemapKernel<<<(int)(num_box / 128) + 1, 128, 0, stream>>>(
        device_box_info, mbox_loc, mbox_priorbox, num_box);

    // find argmax: [class, max_prob]
    ArgmaxKernel<<<(int)(num_box / 128) + 1, 128, 0, stream>>>(
        device_box_info, mbox_conf, num_box, param.mConfidenceThreshold);

    // odd even sort
    int* device_sort_index;
    cudaMallocManaged(&device_sort_index, num_box * 4);

    for (int i = 0; i < num_box; i++) {
        device_sort_index[i] = i;
    }
    for (int i = 0; i < num_box; i++) {
        SortKernel<<<(int)(num_box / 2 / 128) + 1, 128, 0, stream>>>(
            device_box_info, device_sort_index, num_box, i % 2);
    }
    cudaDeviceSynchronize();

    int output_index=0;
    for (int i = 0; i < param.mNMSTopK; i++) {
        if (device_sort_index[i]==-1) {
            continue;
        }
        int total_size=param.mNMSTopK-i;
        NMSKernel<<<(int)( total_size/ 128) + 1, 128, 0, stream>>>(
            dst+output_index*7, device_box_info, device_sort_index+i, total_size, param.mNMSThreshold);
        cudaDeviceSynchronize();
        
        output_index++;
        if (output_index >= param.mKeepTopK) {
            break;
        }
    }
}
