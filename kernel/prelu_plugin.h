// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"
#include "prelu_param.h"

extern void PReLU(
    float* dst, const float* src, struct PReLUParam param, float* slope_weights,
    void* workspace, cudaStream_t);

using namespace nvinfer1;

class PReLUPlugin : public MyPlugin {
   public:
    PReLUPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "channel_shared") {
                mParam.mChannelShared = *((int*)field.data);
            }
            if (std::string(field.name) == "slope_weights") {
                mSlopeWeights = (float*)((Weights*)field.data)->values;
            }
        }
    }
    PReLUPlugin(const void* data, size_t length) {
        mParam = *(struct PReLUParam*)data;
        mSlopeWeights = (float*)malloc(mParam.mSlopeWeightsCount * 4);
        memcpy(
            mSlopeWeights, (char*)data + sizeof(mParam),
            mParam.mSlopeWeightsCount * 4);
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
        return mParam.mSlopeWeightsCount * 4;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        std::cout << *this;
        PReLU(dst, src, mParam, mSlopeWeights, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam) + mParam.mSlopeWeightsCount * 4;
    }

    void serialize(void* buffer) const noexcept override {
        *((struct PReLUParam*)buffer) = mParam;
        memcpy(
            (char*)buffer + sizeof(mParam), mSlopeWeights,
            mParam.mSlopeWeightsCount * 4);
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mParam.mChannel = dims.d[0];
        mParam.mTotalSize = std::accumulate(
            dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
        if (mParam.mChannelShared == 0) {
            mParam.mSlopeWeightsCount = mParam.mChannel;
        } else {
            mParam.mSlopeWeightsCount = 1;
        }
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }

    const char* getPluginType() const noexcept override { return "PRELU"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PReLUPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const PReLUPlugin& c) {
        os << " mChannelShared: " << c.mParam.mChannelShared
           << " mChannel: " << c.mParam.mChannel
           << " mTotalSize: " << c.mParam.mTotalSize
           << " mSlopWeightsCount: " << c.mParam.mSlopeWeightsCount
           << std::endl;

        return os;
    }

   private:
    PReLUParam mParam;
    float* mSlopeWeights;
};
