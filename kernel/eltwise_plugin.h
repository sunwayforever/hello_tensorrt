// 2022-06-14 10:53
#include <assert.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

extern void Eltwise(float*, const float*, const float*, int, int, cudaStream_t);

using namespace nvinfer1;

class EltwisePlugin : public MyPlugin {
   public:
    EltwisePlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "operation") {
                this->mOperation = *(int*)field.data;
            }
        }
    }

    EltwisePlugin(const void* data, size_t length) {
        mInputSize = ((int*)data)[0];
        mOperation = ((int*)data)[1];
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
        const float* src2 = reinterpret_cast<const float*>(inputs[1]);
        Eltwise(dst, src, src2, mInputSize, mOperation, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 2 * 4; }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mInputSize;
        ((int*)buffer)[1] = mOperation;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        assert(nbInput == 2);
        auto dims = in[0].dims;
        mInputSize = std::accumulate(
            dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[0].format == inOut[pos].format &&
               inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const noexcept override { return "ELTWISE"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new EltwisePlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const EltwisePlugin& c) {
        // clang-format off
        return (os
                << " input size: " << c.mInputSize
                << " operation: " << c.mOperation
                << std::endl
        );
        // clang-format on
    }

   private:
    int mInputSize;
    int mOperation;
};
