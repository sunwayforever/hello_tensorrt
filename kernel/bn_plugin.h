// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "bn_param.h"
#include "my_plugin.h"

extern void BN(
    float* dst, const float* src, struct BNParam, float* scale, float* shift,
    void* workspace, cudaStream_t stream);

using namespace nvinfer1;

class BNPlugin : public MyPlugin {
   public:
    BNPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "scale_weights") {
                mScaleWeights = (float*)((Weights*)field.data)->values;
            }
            if (std::string(field.name) == "shift_weights") {
                mShiftWeights = (float*)((Weights*)field.data)->values;
            }
        }
    }

    BNPlugin(const void* data, size_t length) {
        mParam = *(struct BNParam*)data;

        int weightSize = mParam.mChannel * 4;
        float* scale = (float*)malloc(weightSize);
        float* shift = (float*)malloc(weightSize);
        memcpy(scale, (char*)data + sizeof(mParam), weightSize);
        memcpy(shift, (char*)data + sizeof(mParam) + weightSize, weightSize);
        mScaleWeights = scale;
        mShiftWeights = shift;
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return *inputs;
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return mParam.mChannel * 8;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        BN(dst, src, mParam, mScaleWeights, mShiftWeights, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam) + mParam.mChannel * 8;
    }

    void serialize(void* buffer) const noexcept override {
        int weightSize = mParam.mChannel * 4;
        *((struct BNParam*)buffer) = mParam;
        memcpy((char*)buffer + sizeof(mParam), mScaleWeights, weightSize);
        memcpy(
            (char*)buffer + sizeof(mParam) + weightSize, mShiftWeights,
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

    const char* getPluginType() const noexcept override { return "BN"; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new BNPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const BNPlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mParam.mChannel
                << " h: " << c.mParam.mH
                << " w: " << c.mParam.mW
                << std::endl
        );
        // clang-format on
    }

   private:
    BNParam mParam;
    float* mScaleWeights;
    float* mShiftWeights;
};
