// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

using namespace nvinfer1;

extern void Normalize(
    float* dst, const float* src, int channel, int h, int w, int across_spatial,
    int channel_shared, float eps, float* scale_weights, void* workspace,
    cudaStream_t);

class NormalizePlugin : public IPluginV2IOExt {
   public:
    NormalizePlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "across_spatial") {
                this->mAcrossSpatial = *((int*)field.data);
            }
            if (std::string(field.name) == "channel_shared") {
                this->mChannelShared = *((int*)field.data);
            }
            if (std::string(field.name) == "eps") {
                this->mEps = *((float*)field.data);
            }
            if (std::string(field.name) == "scale_weights") {
                this->mScaleWeights = *(Weights*)field.data;
            }
        }
    }

    NormalizePlugin(const void* data, size_t length) {
        mChannel = ((int*)data)[0];
        mH = ((int*)data)[1];
        mW = ((int*)data)[2];
        mAcrossSpatial = ((int*)data)[3];
        mChannelShared = ((float*)data)[4];
        mEps = ((float*)data)[5];
        float* scale = (float*)malloc(mChannel * 4);
        memcpy(scale, ((int*)data) + 6, mChannel * 4);
        mScaleWeights = Weights{
            .type = DataType::kFLOAT,
            .values = scale,
            .count = mChannel,
        };
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return *inputs;
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return (mChannel + mH * mW) * 4;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        Normalize(
            dst, src, mChannel, mH, mW, mAcrossSpatial, mChannelShared, mEps,
            (float*)mScaleWeights.values, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (6 + mScaleWeights.count) * 4;
    }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mChannel;
        ((int*)buffer)[1] = mH;
        ((int*)buffer)[2] = mW;
        ((int*)buffer)[3] = mAcrossSpatial;
        ((int*)buffer)[4] = mChannelShared;
        ((float*)buffer)[5] = mEps;
        memcpy(
            ((int*)buffer) + 6, mScaleWeights.values, mScaleWeights.count * 4);
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
    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override { return "NORMALIZE"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new NormalizePlugin(*this);
        return plugin;
    }
    void setPluginNamespace(const char* libNamespace) noexcept override {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }
    bool isOutputBroadcastAcrossBatch(
        int outputIndex, const bool* inputIsBroadcasted,
        int nbInputs) const noexcept override {
        return false;
    }
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override {
        return false;
    }

    friend std::ostream& operator<<(
        std::ostream& os, const NormalizePlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " across_spatial: " << c.mAcrossSpatial
                << " channel_shared: " << c.mChannelShared
                << " eps: " << c.mEps
                << " scale weights: " << c.mScaleWeights.count
                << std::endl
        );
        // clang-format on
    }

   private:
    int mChannel;
    int mH;
    int mW;
    Weights mScaleWeights;
    int mAcrossSpatial;
    float mEps;
    float mChannelShared;
    std::string mNamespace;
};
