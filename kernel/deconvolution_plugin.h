// 2022-06-14 10:53
#ifndef DECONVOLUTION_PLUGIN_H
#define DECONVOLUTION_PLUGIN_H

#include <assert.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "convolution_plugin.h"
#include "my_plugin.h"

using namespace nvinfer1;

float* InflateDeconvolutionInput(
    const float* input, struct ConvolutionParam param, cudaStream_t stream);

class DeconvolutionPlugin : public ConvolutionPlugin {
   public:
    DeconvolutionPlugin(const PluginFieldCollection fc)
        : ConvolutionPlugin(fc) {}

    DeconvolutionPlugin(const void* data, size_t length)
        : ConvolutionPlugin(data, length) {}

    const char* getPluginType() const noexcept override {
        return "DECONVOLUTION";
    }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new DeconvolutionPlugin(*this);
        return plugin;
    }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        // input=x, output=y
        // (y+2p-k)/s+1=x -> x=?
        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = mParam.mOutputChannel;
        outputDims.d[1] = (h - 1) * mParam.mStrideH -
                          (2 * mParam.mPaddingH -
                           (mParam.mDilationH * (mParam.mKernelH - 1) + 1));
        outputDims.d[2] = (w - 1) * mParam.mStrideW -
                          (2 * mParam.mPaddingW -
                           (mParam.mDilationW * (mParam.mKernelW - 1) + 1));
        return outputDims;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        // input=x, output=y
        // (x+2p-k)/s+1=y -> p=?
        auto inDims = in[0].dims;
        auto outDims = out[0].dims;

        int inputH = (inDims.d[1] - 1) * mParam.mStrideH + 1;
        int inputW = (inDims.d[2] - 1) * mParam.mStrideW + 1;

        int outputH = outDims.d[1];
        int outputW = outDims.d[2];

        mParam.mPaddingH = (outputH - 1) +
                           (mParam.mDilationH * (mParam.mKernelH - 1) + 1) -
                           inputH;
        mParam.mPaddingH /= 2;

        mParam.mPaddingW = (outputW - 1) +
                           (mParam.mDilationW * (mParam.mKernelW - 1) + 1) -
                           inputW;
        mParam.mPaddingW /= 2;

        mParam.mInputChannel = inDims.d[0];
        mParam.mH = inDims.d[1];
        mParam.mW = inDims.d[2];

        // NOTE: kernel 需要做两个变换:
        // 1. IOHW -> OIHW
        // 2. HW 需要中心旋转
        float* kernelWeights = (float*)malloc(mParam.mKernelWeightsSize * 4);
        for (int o = 0; o < mParam.mOutputChannel; o++) {
            for (int i = 0; i < mParam.mInputChannel; i++) {
                for (int h = 0; h < mParam.mKernelH; h++) {
                    for (int w = 0; w < mParam.mKernelW; w++) {
                        kernelWeights
                            [o * mParam.mInputChannel * mParam.mKernelH *
                                 mParam.mKernelW +
                             i * mParam.mKernelH * mParam.mKernelW +
                             h * mParam.mKernelW + w] = ((float*)mKernelWeights)
                                [i * mParam.mOutputChannel * mParam.mKernelH *
                                     mParam.mKernelW +
                                 o * mParam.mKernelH * mParam.mKernelW +
                                 (mParam.mKernelH - h - 1) * mParam.mKernelW +
                                 (mParam.mKernelW - w - 1)];
                    }
                }
            }
        }
        mKernelWeights = kernelWeights;

        mParam.mOrigH = mParam.mH;
        mParam.mOrigW = mParam.mW;
        mParam.mOrigStrideH = mParam.mStrideH;
        mParam.mOrigStrideW = mParam.mStrideW;

        mParam.mH = (mParam.mH - 1) * mParam.mStrideH + 1;
        mParam.mW = (mParam.mW - 1) * mParam.mStrideW + 1;

        mParam.mStrideH = 1;
        mParam.mStrideW = 1;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        std::cout << *this;
        // NOTE: deconv 还有一种等价的计算方法是: 针对 input 中 CHW 的每一个点,
        // 与 Chw 的 kernel 相乘并求和, 得到 hw 大小的数据, 平铺到输出的一个
        // feature map 中

        // NOTE: deconv 的 stride 是指 input 需要`膨胀`的系数, 实际卷积时的
        // stride 固定为 1
        float* input =
            InflateDeconvolutionInput((const float*)inputs[0], mParam, stream);

        return ConvolutionPlugin::enqueue(
            batchSize, std::array<void*, 1>{input}.data(), outputs, workspace,
            stream);
    }

   private:
};
#endif
