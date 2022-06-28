// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void Scale(
    float*, const float*, int, int, int, float*, float*, cudaStream_t);

using namespace nvinfer1;

class ScalePlugin : public IPluginV2IOExt {
   public:
    ScalePlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "scale_weights") {
                this->mScaleWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "bias_weights") {
                this->mBiasWeights = *(Weights*)field.data;
            }
        }
    }

    ScalePlugin(const void* data, size_t length) {
        mChannel = ((int*)data)[0];
        mH = ((int*)data)[1];
        mW = ((int*)data)[2];
        int sc = ((int*)data)[3];
        int bc = ((int*)data)[4];

        float* scale = (float*)malloc(sc * 4);
        float* bias = (float*)malloc(bc * 4);

        memcpy(scale, ((int*)data) + 5, sc * 4);
        memcpy(bias, ((int*)data) + 5 + sc, bc * 4);

        mScaleWeights = Weights{
            .type = DataType::kFLOAT,
            .values = scale,
            .count = sc,
        };

        mBiasWeights = Weights{
            .type = DataType::kFLOAT,
            .values = bias,
            .count = bc,
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
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        Scale(
            dst, src, mChannel, mH, mW, (float*)mScaleWeights.values,
            mBiasWeights.count == 0 ? NULL : (float*)mBiasWeights.values,
            stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (5 + mScaleWeights.count + mBiasWeights.count) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mChannel;
        ((int*)buffer)[1] = mH;
        ((int*)buffer)[2] = mW;
        ((int*)buffer)[3] = mScaleWeights.count;
        ((int*)buffer)[4] = mBiasWeights.count;
        memcpy(
            ((int*)buffer) + 5, mScaleWeights.values, mScaleWeights.count * 4);
        memcpy(
            ((int*)buffer) + 5 + mScaleWeights.count, mBiasWeights.values,
            mBiasWeights.count * 4);
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

    const char* getPluginType() const noexcept override { return "SCALE"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new ScalePlugin(*this);
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

    friend std::ostream& operator<<(std::ostream& os, const ScalePlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " mean size: " << c.mScaleWeights.count
                << " var size: " << c.mBiasWeights.count
                << std::endl
        );
        // clang-format on
    }

   private:
    int mChannel;
    int mH;
    int mW;
    Weights mScaleWeights;
    Weights mBiasWeights;
    std::string mNamespace;
};

class ScalePluginCreator : public IPluginCreator {
   public:
    const char* getPluginName() const noexcept override { return "SCALE"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override {
        return &mFieldCollection;
    }
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc) noexcept override {
        auto* plugin = new ScalePlugin(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }
    IPluginV2* deserializePlugin(
        const char* name, const void* serialData,
        size_t serialLength) noexcept override {
        auto* plugin = new ScalePlugin(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }
    void setPluginNamespace(const char* libNamespace) noexcept override {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

   private:
    std::string mNamespace;
    std::string mPluginName;
    PluginFieldCollection mFieldCollection{0, nullptr};
};
