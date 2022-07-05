// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "batch_norm_param.h"

extern void BatchNorm(
    float* dst, const float* src, struct BatchNormParam, float* mean,
    float* var, void* workspace, cudaStream_t stream);

using namespace nvinfer1;

class BatchNormPlugin : public IPluginV2IOExt {
   public:
    BatchNormPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "eps") {
                mParam.mEps = *((float*)field.data);
            }
            if (std::string(field.name) == "moving_average") {
                mParam.mMovingAverage = *((float*)field.data);
            }
            if (std::string(field.name) == "mean_weights") {
                mMeanWeights = (float*)((Weights*)field.data)->values;
            }
            if (std::string(field.name) == "var_weights") {
                mVarWeights = (float*)((Weights*)field.data)->values;
            }
        }
    }

    BatchNormPlugin(const void* data, size_t length) {
        mParam = *(struct BatchNormParam*)data;

        int weightSize = mParam.mChannel * 4;
        float* mean = (float*)malloc(weightSize);
        float* var = (float*)malloc(weightSize);
        memcpy(mean, (char*)data + sizeof(mParam), weightSize);
        memcpy(var, (char*)data + sizeof(mParam) + weightSize, weightSize);
        mMeanWeights = mean;
        mVarWeights = var;
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
        return mParam.mChannel * 8;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        BatchNorm(
            dst, src, mParam, mMeanWeights, mVarWeights, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam) + mParam.mChannel * 8;
    }

    void serialize(void* buffer) const noexcept override {
        int weightSize = mParam.mChannel * 4;
        *((struct BatchNormParam*)buffer) = mParam;
        memcpy((char*)buffer + sizeof(mParam), mMeanWeights, weightSize);
        memcpy(
            (char*)buffer + sizeof(mParam) + weightSize, mVarWeights,
            weightSize);
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mParam.mChannel = dims.d[0];
        mParam.mH = dims.d[1];
        mParam.mW = dims.d[2];
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
                << " channel: " << c.mParam.mChannel
                << " h: " << c.mParam.mH
                << " w: " << c.mParam.mW
                << " eps: " << c.mParam.mEps
                << " moving average: " << c.mParam.mMovingAverage
                << std::endl
        );
        // clang-format on
    }

   private:
    BatchNormParam mParam;
    float* mMeanWeights;
    float* mVarWeights;
    std::string mNamespace;
};
