// 2022-06-14 10:53
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

using namespace nvinfer1;

extern void LRN(
    float* dst, const float* src, int channel, int h, int w, int local_size,
    float alpha, float beta, cudaStream_t);

class LRNPlugin : public MyPlugin {
   public:
    LRNPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "local_size") {
                this->mLocalSize = *((int*)field.data);
            } else if (std::string(field.name) == "alpha") {
                this->mAlpha = *((float*)field.data);
            } else if (std::string(field.name) == "beta") {
                this->mBeta = *((float*)field.data);
            }
        }
    }

    LRNPlugin(const void* data, size_t length) {
        mChannel = ((int*)data)[0];
        mH = ((int*)data)[1];
        mW = ((int*)data)[2];
        mLocalSize = ((int*)data)[3];
        mAlpha = ((float*)data)[4];
        mBeta = ((float*)data)[5];
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return *inputs;
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        LRN(dst, src, mChannel, mH, mW, mLocalSize, mAlpha, mBeta, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 24; }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mChannel;
        ((int*)buffer)[1] = mH;
        ((int*)buffer)[2] = mW;
        ((int*)buffer)[3] = mLocalSize;
        ((float*)buffer)[4] = mAlpha;
        ((float*)buffer)[5] = mBeta;
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mChannel = dims.d[0];
        mH = dims.d[1];
        mW = dims.d[2];
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].format == TensorFormat::kLINEAR &&
               inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const noexcept override { return "LRN"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new LRNPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const LRNPlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " local_size: " << c.mLocalSize
                << " alpha: " << c.mAlpha
                << " beta: " << c.mBeta
                << std::endl
        );
        // clang-format on
    }

   private:
    int mChannel;
    int mH;
    int mW;
    int mLocalSize;
    float mAlpha;
    float mBeta;
};
