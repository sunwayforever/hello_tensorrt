// 2022-06-14 10:53
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

extern void Relu(float*, float*, int, cudaStream_t);

using namespace nvinfer1;

class ReluPlugin : public MyPlugin {
   public:
    ReluPlugin(const PluginFieldCollection fc) {}
    ReluPlugin(const void* data, size_t length) {
        mInputSize = ((int*)data)[0];
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
        Relu(dst, const_cast<float*>(src), mInputSize, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 4; }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mInputSize;
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mInputSize = std::accumulate(
            dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }

    const char* getPluginType() const noexcept override { return "RELU"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new ReluPlugin(*this);
        return plugin;
    }

   private:
    int mInputSize;
};
