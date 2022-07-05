// 2022-06-14 10:53
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void PriorBox(
    float*, int h, int w, int image_h, int image_w, float offset, float step,
    std::vector<float> min_size, std::vector<float> max_size,
    std::vector<float> aspect_ration, cudaStream_t);

using namespace nvinfer1;

class PriorBox2Plugin : public IPluginV2IOExt {
   public:
    PriorBox2Plugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "min_size") {
                for (int i = 0; i < field.length; i++) {
                    mMinSize.push_back(((float*)field.data)[i]);
                }
            }
            if (std::string(field.name) == "max_size") {
                for (int i = 0; i < field.length; i++) {
                    mMaxSize.push_back(((float*)field.data)[i]);
                }
            }
            if (std::string(field.name) == "aspect_ratio") {
                for (int i = 0; i < field.length; i++) {
                    mAspectRatio.push_back(((float*)field.data)[i]);
                }
            }
            if (std::string(field.name) == "flip") {
                mFlip = *(int*)field.data;
            }
            if (std::string(field.name) == "offset") {
                mOffset = *(float*)field.data;
            }
            if (std::string(field.name) == "step") {
                mStep = *(float*)field.data;
            }
        }
    }
    PriorBox2Plugin(const void* data, size_t length) {
        mH = ((int*)data)[0];
        mW = ((int*)data)[1];
        mImageH = ((int*)data)[2];
        mImageW = ((int*)data)[3];
        mFlip = ((int*)data)[4];
        mOffset = ((float*)data)[5];
        mStep = ((float*)data)[6];
        int minSize = ((int*)data)[7];
        int maxSize = ((int*)data)[8];
        int aspectRatioSize = ((int*)data)[9];
        mMinSize = std::vector<float>(minSize);
        memcpy(mMinSize.data(), (float*)data + 10, minSize * 4);
        mMaxSize = std::vector<float>(maxSize);
        memcpy(mMaxSize.data(), (float*)data + 10 + minSize, maxSize * 4);
        mAspectRatio = std::vector<float>(aspectRatioSize);
        memcpy(
            mAspectRatio.data(), (float*)data + 10 + minSize + maxSize,
            aspectRatioSize * 4);
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int h = inputs[0].d[1];
        int w = inputs[0].d[2];
        mH = h;
        mW = w;
        int boxCount =
            (mAspectRatio.size() * (mFlip ? 2 : 1) + 1) * mMinSize.size() +
            mMaxSize.size();
        return Dims3(2, h * w * boxCount * 4, 1);
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
        PriorBox(
            dst, mH, mW, mImageH, mImageW, mOffset, mStep, mMinSize, mMaxSize,
            mAspectRatio, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (10 + mMinSize.size() + mMaxSize.size() + mAspectRatio.size()) *
               4;
    }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mH;
        ((int*)buffer)[1] = mW;
        ((int*)buffer)[2] = mImageH;
        ((int*)buffer)[3] = mImageW;
        ((int*)buffer)[4] = mFlip;
        ((float*)buffer)[5] = mOffset;
        ((float*)buffer)[6] = mStep;
        ((int*)buffer)[7] = mMinSize.size();
        ((int*)buffer)[8] = mMaxSize.size();
        ((int*)buffer)[9] = mAspectRatio.size();
        memcpy((float*)buffer + 10, mMinSize.data(), mMinSize.size() * 4);
        memcpy(
            (float*)buffer + 10 + mMinSize.size(), mMaxSize.data(),
            mMaxSize.size() * 4);
        memcpy(
            (float*)buffer + 10 + mMinSize.size() + mMaxSize.size(),
            mAspectRatio.data(), mAspectRatio.size() * 4);
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mH = dims.d[1];
        mW = dims.d[2];
        mImageH = in[1].dims.d[1];
        mImageW = in[1].dims.d[2];
        auto tmpAspectRatio = std::vector<float>();
        tmpAspectRatio.push_back(1.0f);
        for (int i = 0; i < mAspectRatio.size(); i++) {
            tmpAspectRatio.push_back(mAspectRatio[i]);
            if (mFlip) {
                tmpAspectRatio.push_back(1.0f / mAspectRatio[i]);
            }
        }
        mAspectRatio = tmpAspectRatio;
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

    const char* getPluginType() const noexcept override { return "PRIOR_BOX"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PriorBox2Plugin(*this);
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
        std::ostream& os, const PriorBox2Plugin& c) {
        os << " h: " << c.mH << " w: " << c.mW << " image_h: " << c.mImageH
           << " image_w: " << c.mImageW;

        os << " mMinSize: " << c.mMinSize.size() << ":";
        for (int i = 0; i < c.mMinSize.size(); i++) {
            os << c.mMinSize[i] << " ";
        }

        os << " mMaxSize: " << c.mMaxSize.size() << ":";
        for (int i = 0; i < c.mMaxSize.size(); i++) {
            os << c.mMaxSize[i] << " ";
        }

        os << " mAspectRatio: " << c.mAspectRatio.size() << ":";
        for (int i = 0; i < c.mAspectRatio.size(); i++) {
            os << c.mAspectRatio[i] << " ";
        }

        os << " mFlip: " << c.mFlip << " mStep: " << c.mStep
           << " mOffset: " << c.mOffset << std::endl;
        return os;
    }

   private:
    int mH;
    int mW;
    int mImageH;
    int mImageW;
    int mFlip;
    float mOffset;
    float mStep;
    std::vector<float> mMinSize;
    std::vector<float> mMaxSize;
    std::vector<float> mAspectRatio;
    std::string mNamespace;
};
