// 2022-06-14 10:53
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"

extern void Softmax(float*, float*, int*, int, cudaStream_t);

using namespace nvinfer1;

class SoftmaxPlugin : public MyPlugin {
   public:
    SoftmaxPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "axis") {
                this->mAxis = *((int*)field.data);
            }
        }
    }
    SoftmaxPlugin(const void* data, size_t length) {
        mAxis = ((int*)data)[0];
        mNewAxis = ((int*)data)[1];
        mDims[0] = ((int*)data)[2];
        mDims[1] = ((int*)data)[3];
        mDims[2] = ((int*)data)[4];
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
        Softmax(dst, const_cast<float*>(src), mDims, mNewAxis, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 20; }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mAxis;
        ((int*)buffer)[1] = mNewAxis;
        ((int*)buffer)[2] = mDims[0];
        ((int*)buffer)[3] = mDims[1];
        ((int*)buffer)[4] = mDims[2];
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        for (int i = 0; i < dims.nbDims; i++) {
            mDims[i + 3 - dims.nbDims] = dims.d[i];
        }
        if (mAxis == -1) {
            mAxis = dims.nbDims;
        }
        mNewAxis = mAxis - 1 + 3 - dims.nbDims;
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }

    const char* getPluginType() const noexcept override { return "SOFTMAX"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new SoftmaxPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const SoftmaxPlugin& c) {
        // clang-format off
        return (os
                << " axis: " << c.mAxis
                << " new axis: " << c.mNewAxis
                << " dim: " << c.mDims[0] << " " << c.mDims[1] << " " << c.mDims[2]
                << std::endl
        );
        // clang-format on
    }

   private:
    int mAxis;
    int mNewAxis;
    int mDims[3] = {1, 1, 1};
};
