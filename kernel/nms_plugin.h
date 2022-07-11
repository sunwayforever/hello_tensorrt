// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "my_plugin.h"
#include "nms_param.h"

extern void NMS(
    float* dst, const float* mbox_loc, const float* mbox_conf,
    const float* mbox_priorbox, struct NMSParam, void* workspace,
    cudaStream_t stream);

using namespace nvinfer1;

class NMSPlugin : public MyPlugin {
   public:
    NMSPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "num_classes") {
                mParam.mNumClasses = *(int*)field.data;
            } else if (std::string(field.name) == "keep_top_k") {
                mParam.mKeepTopK = *(int*)field.data;
            } else if (std::string(field.name) == "confidence_threshold") {
                mParam.mConfidenceThreshold = *(float*)field.data;
            } else if (std::string(field.name) == "nms_threshold") {
                mParam.mNMSThreshold = *(float*)field.data;
            } else if (std::string(field.name) == "nms_top_k") {
                mParam.mNMSTopK = *(int*)field.data;
            }
        }
    }
    NMSPlugin(const void* data, size_t length) {
        mParam = *(struct NMSParam*)data;
    }

   public:
    int getNbOutputs() const noexcept override { return 2; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        Dims outputDims;
        if (index == 0) {
            return Dims3{1, mParam.mKeepTopK, 7};
        } else {
            return Dims3{1, 1, 1};
        }
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        // for input_dims, input_mul and output_mul
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        std::cout << *this;
        const float* mboxLoc = (const float*)inputs[0];
        const float* mboxConf = (const float*)inputs[1];
        const float* mboxPriorbox = (const float*)inputs[2];
        NMS(dst, mboxLoc, mboxConf, mboxPriorbox, mParam, workspace, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(mParam);
    }

    void serialize(void* buffer) const noexcept override {
        *(struct NMSParam*)buffer = mParam;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        // mbox_loc[Float(-2,34928,1,1)]  34928 / 4 = 8732
        // mbox_conf_flatten[Float(-2,183372,1,1)] -> 183372 / 21 = 8732
        // mbox_priorbox[Float(-2,2,34928,1)]
        // 共 8732 个 box
        auto inputDims = in[0].dims;
        mParam.mNumBox = inputDims.d[0] / 4;
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        return inOut[pos].format == TensorFormat::kLINEAR &&
               inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const noexcept override { return "NMS"; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new NMSPlugin(*this);
        return plugin;
    }

    friend std::ostream& operator<<(std::ostream& os, const NMSPlugin& c) {
        os << " num_classes: " << c.mParam.mNumClasses;
        os << " keep_top_k: " << c.mParam.mKeepTopK;
        os << " confidence_threshold: " << c.mParam.mConfidenceThreshold;
        os << " nms_top_k: " << c.mParam.mNMSTopK;
        os << " nms_threshold: " << c.mParam.mNMSThreshold;
        os << " num_box: " << c.mParam.mNumBox;
        os << std::endl;
        return os;
    }

   private:
    NMSParam mParam;
};
