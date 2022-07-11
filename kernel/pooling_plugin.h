// 2022-06-14 10:53
#include <assert.h>
#include <unistd.h>

#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"
#include "pooling_param.h"

extern void Pooling(
    float* dst, const float* src, struct PoolingParam param,
    cudaStream_t stream);

using namespace nvinfer1;

// NOTE: caffe pooling is 2d pooling
class PoolingPlugin : public MyPlugin {
   public:
    PoolingPlugin(const PluginFieldCollection fc) {
        mParam.mMethod = 0, mParam.mPaddingH = 0, mParam.mPaddingW = 0;
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "method") {
                this->mParam.mMethod = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_h") {
                this->mParam.mKernelH = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_w") {
                this->mParam.mKernelW = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_h") {
                this->mParam.mStrideH = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_w") {
                this->mParam.mStrideW = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_h") {
                this->mParam.mPaddingH = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_w") {
                this->mParam.mPaddingW = *((int*)field.data);
            }
            if (std::string(field.name) == "global_pooling") {
                this->mParam.mGlobalPooling = *((int*)field.data);
            }
        }
    }

    PoolingPlugin(const void* data, size_t length) {
        mParam = *(struct PoolingParam*)data;
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    // NOTE: NCHW format
    static int ceil(int a, int b) {
        assert(a >= 0);
        assert(b > 0);
        return a / b + (a % b > 0);
    }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = channel;
        // NOTE: caffe pooling padding is always symmetric
        // NOTE: `ceil` for pooling by default
        if (mParam.mGlobalPooling == 1) {
            outputDims.d[1] = 1;
            outputDims.d[2] = 1;
        } else {
            outputDims.d[1] = ceil(
                                  h + 2 * mParam.mPaddingH - mParam.mKernelH,
                                  mParam.mStrideH) +
                              1;
            outputDims.d[2] = ceil(
                                  w + 2 * mParam.mPaddingW - mParam.mKernelW,
                                  mParam.mStrideW) +
                              1;
        }
        return outputDims;
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        std::cout << *this;
        Pooling(dst, src, mParam, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam);
    }

    void serialize(void* buffer) const noexcept override {
        *(struct PoolingParam*)buffer = mParam;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mParam.mChannel = dims.d[0];
        mParam.mH = dims.d[1];
        mParam.mW = dims.d[2];
        if (mParam.mGlobalPooling == 1) {
            mParam.mKernelH = mParam.mH;
            mParam.mKernelW = mParam.mW;
        }
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].format == TensorFormat::kLINEAR &&
               inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const noexcept override { return "POOLING"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PoolingPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const PoolingPlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mParam.mChannel
                << " h: " << c.mParam.mH
                << " w: " << c.mParam.mW
                << " kernel: " << c.mParam.mKernelH << " " << c.mParam.mKernelW
                << " stride: " << c.mParam.mStrideH << " " << c.mParam.mStrideW
                << " pad: " << c.mParam.mPaddingH << " " << c.mParam.mPaddingW
                << " method: " << c.mParam.mMethod
                << std::endl
        );
        // clang-format on
    }

   private:
    struct PoolingParam mParam;
};
