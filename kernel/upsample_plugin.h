// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"
#include "upsample_param.h"

extern void Upsample(
    float* dst, const float* src, const float* mask, struct UpsampleParam param,
    void* workspace, cudaStream_t);

using namespace nvinfer1;

class UpsamplePlugin : public MyPlugin {
   public:
    UpsamplePlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "scale") {
                mParam.mScale = *((int*)field.data);
            }
        }
    }
    UpsamplePlugin(const void* data, size_t length) {
        mParam = *(struct UpsampleParam*)data;
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        Dims inputDim = inputs[0];
        return Dims3{
            inputDim.d[0], inputDim.d[1] * mParam.mScale,
            inputDim.d[2] * mParam.mScale};
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        const float* mask = reinterpret_cast<const float*>(inputs[1]);
        Upsample(dst, src, mask, mParam, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam);
    }

    void serialize(void* buffer) const noexcept override {
        *((struct UpsampleParam*)buffer) = mParam;
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
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }

    const char* getPluginType() const noexcept override { return "UPSAMPLE"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new UpsamplePlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const UpsamplePlugin& c) {
        os << " mScale: " << c.mParam.mScale
           << " mChannel: " << c.mParam.mChannel << " mH: " << c.mParam.mH
           << " mW: " << c.mParam.mW << std::endl;
        return os;
    }

   private:
    UpsampleParam mParam;
};
