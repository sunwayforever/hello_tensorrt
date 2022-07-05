// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void Permute(
    float* dst, const float* src, int nb_dims, int* dims, int* input_mul,
    int* output_mul, void* workspace, cudaStream_t stream);

using namespace nvinfer1;

class PermutePlugin : public IPluginV2IOExt {
   public:
    PermutePlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "order") {
                this->mOrder = (int*)field.data;
            }
        }
    }
    PermutePlugin(const void* data, size_t length) {
        mNbDims = ((int*)data)[0];
        mInputDims = (int*)malloc(mNbDims * 4);
        mInputMul = (int*)malloc(mNbDims * 4);
        mOutputMul = (int*)malloc(mNbDims * 4);
        memcpy(mInputDims, (int*)data + 1, mNbDims * 4);
        memcpy(mInputMul, (int*)data + 1 + mNbDims, mNbDims * 4);
        memcpy(mOutputMul, (int*)data + 1 + mNbDims * 2, mNbDims * 4);
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        Dims outputDims;
        outputDims.nbDims = inputs->nbDims;
        for (int i = 0; i < inputs->nbDims; i++) {
            outputDims.d[i] = inputs->d[mOrder[i]];
        }
        return outputDims;
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        // for input_dims, input_mul and output_mul
        return mNbDims * 3 * 4;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);
        Permute(
            dst, src, mNbDims, mInputDims, mInputMul, mOutputMul, workspace,
            stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (1 + mNbDims * 3) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mNbDims;
        memcpy((int*)buffer + 1, mInputDims, mNbDims * 4);
        memcpy((int*)buffer + 1 + mNbDims, mInputMul, mNbDims * 4);
        memcpy((int*)buffer + 1 + mNbDims * 2, mOutputMul, mNbDims * 4);
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto inputDims = in[0].dims;
        auto outputDims = out[0].dims;

        mNbDims = in[0].dims.nbDims;
        mInputDims = (int*)malloc(mNbDims * 4);
        for (int i = 0; i < mNbDims; i++) {
            mInputDims[i] = inputDims.d[i];
        }
        mInputMul = (int*)malloc(mNbDims * 4);
        mOutputMul = (int*)malloc(mNbDims * 4);
        for (int i = 0; i < mNbDims; i++) {
            mInputMul[i] = 1;
            mOutputMul[i] = 1;
        }
        // NOTE:
        // suppose input dim is [3, 100, 200], order = [1, 2, 0]]
        // then permuted dim is [100, 200, 3],
        // input mul = [20000, 200, 1]
        // output mul = [1, 600, 3]
        for (int i = 0; i < mNbDims; i++) {
            mInputMul[i] = std::accumulate(
                inputDims.d + i + 1, inputDims.d + inputDims.nbDims, 1,
                std::multiplies<int>());
            mOutputMul[mOrder[i]] = std::accumulate(
                outputDims.d + i + 1, outputDims.d + outputDims.nbDims, 1,
                std::multiplies<int>());
        }
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == inOut[0].format;
    }
    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override { return "PERMUTE"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PermutePlugin(*this);
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

    friend std::ostream& operator<<(std::ostream& os, const PermutePlugin& c) {
        os << " input dims: ";
        for (int i = 0; i < c.mNbDims; i++) {
            os << c.mInputDims[i] << " ";
        }
        os << " input mul: ";
        for (int i = 0; i < c.mNbDims; i++) {
            os << c.mInputMul[i] << " ";
        }

        os << " output mul: ";
        for (int i = 0; i < c.mNbDims; i++) {
            os << c.mOutputMul[i] << " ";
        }
        os << std::endl;
        return os;
    }

   private:
    int mNbDims;
    int* mOrder;
    int* mInputDims;
    int* mInputMul;
    int* mOutputMul;
    std::string mNamespace;
};
