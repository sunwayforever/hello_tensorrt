// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void BatchNorm(
    float*, const float*, int, int, int, float, float, float*, float*,
    cudaStream_t);

using namespace nvinfer1;

class BatchNormPlugin : public IPluginV2IOExt {
   public:
    BatchNormPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "eps") {
                this->mEps = *((float*)field.data);
            }
            if (std::string(field.name) == "moving_average") {
                this->mMovingAverage = *((float*)field.data);
            }
            if (std::string(field.name) == "mean_weights") {
                this->mMeanWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "var_weights") {
                this->mVarWeights = *(Weights*)field.data;
            }
        }
    }

    BatchNormPlugin(const void* data, size_t length) {
        mChannel = ((int*)data)[0];
        mH = ((int*)data)[1];
        mW = ((int*)data)[2];
        mEps = ((float*)data)[3];
        mMovingAverage = ((float*)data)[4];
        int mc = ((int*)data)[5];
        int vc = ((int*)data)[6];

        float* mean = (float*)malloc(mc * 4);
        float* var = (float*)malloc(vc * 4);

        memcpy(mean, ((int*)data) + 7, mc * 4);
        memcpy(var, ((int*)data) + 7 + mc, vc * 4);

        mMeanWeights = Weights{
            .type = DataType::kFLOAT,
            .values = mean,
            .count = mc,
        };

        mVarWeights = Weights{
            .type = DataType::kFLOAT,
            .values = var,
            .count = vc,
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
        BatchNorm(
            dst, src, mChannel, mH, mW, mEps, mMovingAverage,
            (float*)mMeanWeights.values, (float*)mVarWeights.values, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (7 + mMeanWeights.count + mVarWeights.count) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mChannel;
        ((int*)buffer)[1] = mH;
        ((int*)buffer)[2] = mW;
        ((float*)buffer)[3] = mEps;
        ((float*)buffer)[4] = mMovingAverage;
        ((int*)buffer)[5] = mMeanWeights.count;
        ((int*)buffer)[6] = mVarWeights.count;
        memcpy(((int*)buffer) + 7, mMeanWeights.values, mMeanWeights.count * 4);
        memcpy(
            ((int*)buffer) + 7 + mMeanWeights.count, mVarWeights.values,
            mVarWeights.count * 4);
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

    const char* getPluginType() const noexcept override { return "BATCH_NORM"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new BatchNormPlugin(*this);
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
        std::ostream& os, const BatchNormPlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " eps: " << c.mEps
                << " moving average: " << c.mMovingAverage
                << " mean size: " << c.mMeanWeights.count
                << " var size: " << c.mVarWeights.count
                << std::endl
        );
        // clang-format on
    }

   private:
    float mEps;
    float mMovingAverage;
    int mChannel;
    int mH;
    int mW;
    Weights mMeanWeights;
    Weights mVarWeights;
    std::string mNamespace;
};

class BatchNormPluginCreator : public IPluginCreator {
   public:
    const char* getPluginName() const noexcept override { return "BATCH_NORM"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override {
        return &mFieldCollection;
    }
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc) noexcept override {
        auto* plugin = new BatchNormPlugin(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }
    IPluginV2* deserializePlugin(
        const char* name, const void* serialData,
        size_t serialLength) noexcept override {
        auto* plugin = new BatchNormPlugin(serialData, serialLength);
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
