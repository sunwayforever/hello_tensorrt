// 2022-06-14 10:53
#ifndef CONVOLUTION_PLUGIN_H
#define CONVOLUTION_PLUGIN_H

#include <assert.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "convolution_param.h"
#include "my_plugin.h"

extern void Convolution(
    void* dst, const void* src, ConvolutionParam param, void* kernel,
    void* bias, void* workspace, cudaStream_t stream);

using namespace nvinfer1;

class ConvolutionPlugin : public MyPlugin {
   public:
    ConvolutionPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "num_output") {
                mParam.mOutputChannel = *((int*)field.data);
            }
            if (std::string(field.name) == "group") {
                mParam.mGroup = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_weights") {
                mKernelWeights =
                    const_cast<void*>(((Weights*)field.data)->values);
                mParam.mKernelWeightsSize = ((Weights*)field.data)->count;
            }
            if (std::string(field.name) == "bias_weights") {
                mBiasWeights =
                    const_cast<void*>(((Weights*)field.data)->values);
                mParam.mBiasWeightsSize = ((Weights*)field.data)->count;
            }
            if (std::string(field.name) == "kernel_h") {
                mParam.mKernelH = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_w") {
                mParam.mKernelW = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_h") {
                mParam.mStrideH = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_w") {
                mParam.mStrideW = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_h") {
                mParam.mPaddingH = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_w") {
                mParam.mPaddingW = *((int*)field.data);
            }
            if (std::string(field.name) == "dilation_h") {
                mParam.mDilationH = *((int*)field.data);
            }
            if (std::string(field.name) == "dilation_w") {
                mParam.mDilationW = *((int*)field.data);
            }
        }
    }

    ConvolutionPlugin(const void* data, size_t length) {
        mParam = *(struct ConvolutionParam*)data;

        int kc = mParam.mKernelWeightsSize;
        int bc = mParam.mBiasWeightsSize;
        int size = mParam.mType == (int)DataType::kFLOAT ? 4 : 1;
        cudaMalloc(&mKernelWeights, kc * size);
        cudaMalloc(&mBiasWeights, bc * size);
        cudaMemcpy(
            mKernelWeights, (char*)data + sizeof(mParam), kc * size,
            cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(
            mBiasWeights, (char*)data + sizeof(mParam) + kc * size, bc * size,
            cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    static int floor(int a, int b) {
        assert(a >= 0);
        assert(b > 0);
        return a / b;
    }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = mParam.mOutputChannel;
        // NOTE: `floor` for convolution
        outputDims.d[1] =
            floor(
                h + 2 * mParam.mPaddingH -
                    (mParam.mDilationH * (mParam.mKernelH - 1) + 1),
                mParam.mStrideH) +
            1;
        outputDims.d[2] =
            floor(
                w + 2 * mParam.mPaddingW -
                    (mParam.mDilationW * (mParam.mKernelW - 1) + 1),
                mParam.mStrideW) +
            1;

        return outputDims;
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        void* dst = outputs[0];
        const void* src = inputs[0];
        Convolution(
            dst, src, mParam, mKernelWeights,
            mParam.mBiasWeightsSize == 0 ? NULL : mBiasWeights, workspace,
            stream);
        return 0;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        mParam.mType = (int)in[0].type;
        mParam.mInputScale = in[0].scale;
        mParam.mOutputScale = out[0].scale;
        auto dims = in[0].dims;
        mParam.mInputChannel = dims.d[0];
        mParam.mH = dims.d[1];
        mParam.mW = dims.d[2];

        if (mParam.mType == (int)DataType::kINT8) {
            int8_t* tmpBiasWeights = (int8_t*)malloc(mParam.mOutputChannel);
            int8_t* tmpKernelWeights = (int8_t*)malloc(
                (mParam.mInputChannel * mParam.mOutputChannel *
                 mParam.mKernelH * mParam.mKernelW));
            float kernel_max = ((float*)mKernelWeights)[0];
            float kernel_min = ((float*)mKernelWeights)[0];
            for (int i = 0; i < mParam.mKernelWeightsSize; i++) {
                if (((float*)mKernelWeights)[i] > kernel_max) {
                    kernel_max = ((float*)mKernelWeights)[i];
                }
                if (((float*)mKernelWeights)[i] < kernel_min) {
                    kernel_min = ((float*)mKernelWeights)[i];
                }
            }

            mParam.mKernelScale =
                (float)std::max(std::fabs(kernel_max), std::fabs(kernel_min)) /
                127;

            for (int i = 0; i < mParam.mKernelWeightsSize; i++) {
                tmpKernelWeights[i] = (int8_t)(
                    ((float*)mKernelWeights)[i] / mParam.mKernelScale);  // Q
            }

            if (mParam.mBiasWeightsSize != 0) {
                for (int i = 0; i < mParam.mBiasWeightsSize; i++) {
                    tmpBiasWeights[i] = (int8_t)(
                        ((float*)mBiasWeights)[i] /
                        (mParam.mKernelScale * mParam.mInputScale));  // Q
                }
            }
            mKernelWeights = tmpKernelWeights;
            mBiasWeights = tmpBiasWeights;
        }
    }

    size_t getSerializationSize() const noexcept override {
        int size = mParam.mType == (int)DataType::kFLOAT ? 4 : 1;
        return sizeof(mParam) +
               (mParam.mKernelWeightsSize + mParam.mBiasWeightsSize) * size;
    }

    void serialize(void* buffer) const noexcept override {
        *(struct ConvolutionParam*)buffer = mParam;
        int size = mParam.mType == (int)DataType::kFLOAT ? 4 : 1;
        memcpy(
            (char*)buffer + sizeof(mParam), mKernelWeights,
            mParam.mKernelWeightsSize * size);
        memcpy(
            (char*)buffer + sizeof(mParam) + mParam.mKernelWeightsSize * size,
            mBiasWeights, mParam.mBiasWeightsSize * size);
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        // std::cout << (int)inOut[pos].format << " " <<
        // (int)inOut[pos].type
        //           << " " << (int)inOut[0].type << std::endl;
        return inOut[pos].format == TensorFormat::kLINEAR &&
               (inOut[pos].type == DataType::kFLOAT ||
                inOut[pos].type == DataType::kINT8) &&
               inOut[pos].type == inOut[0].type;
    }

    const char* getPluginType() const noexcept override {
        return "CONVOLUTION";
    }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new ConvolutionPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(
        std::ostream& os, const ConvolutionPlugin& c) {
        return (
            os << " input channel: " << c.mParam.mInputChannel
               << " output channel: " << c.mParam.mOutputChannel
               << " group: " << c.mParam.mGroup << " h: " << c.mParam.mH
               << " w: " << c.mParam.mW << " kernel: " << c.mParam.mKernelH
               << " " << c.mParam.mKernelW << " stride: " << c.mParam.mStrideH
               << " " << c.mParam.mStrideW << " pad: " << c.mParam.mPaddingH
               << " " << c.mParam.mPaddingW << " type: " << c.mParam.mType
               << " scale: " << c.mParam.mInputScale << " "
               << c.mParam.mOutputScale << " " << c.mParam.mKernelScale
               << " dilation: " << c.mParam.mDilationH << " "
               << c.mParam.mDilationW << " "
               << " kernel size: " << c.mParam.mKernelWeightsSize << " "
               << " bias size: " << c.mParam.mBiasWeightsSize << " "
               << std::endl);
    }

   protected:
    ConvolutionParam mParam;

    void* mKernelWeights;
    void* mBiasWeights;
};
#endif
