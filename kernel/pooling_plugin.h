// 2022-06-14 10:53
#include <assert.h>
#include <unistd.h>

#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void Pooling(
    float* dst, const float* src, int channel, int h, int w, int method,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, cudaStream_t);

using namespace nvinfer1;

// NOTE: caffe pooling is 2d pooling
class PoolingPlugin : public IPluginV2IOExt {
   public:
    PoolingPlugin(const PluginFieldCollection fc)
        : mMethod(0), mPadH(0), mPadW(0) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "method") {
                this->mMethod = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_h") {
                this->mKernelH = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_w") {
                this->mKernelW = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_h") {
                this->mStrideH = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_w") {
                this->mStrideW = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_h") {
                this->mPadH = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_w") {
                this->mPadW = *((int*)field.data);
            }
            if (std::string(field.name) == "global_pooling") {
                this->mGlobalPooling = *((int*)field.data);
            }
        }
    }

    PoolingPlugin(const void* data, size_t length) {
        mMethod = ((int*)data)[0];
        mKernelW = ((int*)data)[1];
        mKernelH = ((int*)data)[2];
        mStrideW = ((int*)data)[3];
        mStrideH = ((int*)data)[4];
        mPadW = ((int*)data)[5];
        mPadH = ((int*)data)[6];
        mChannel = ((int*)data)[7];
        mH = ((int*)data)[8];
        mW = ((int*)data)[9];
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    // NOTE: NCHW format
    static int ceil(int a, int b) {
        assert(a >= 0);
        assert(b > 0);
        return a / b + (a % b > 0);
    }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = channel;
        // NOTE: caffe pooling padding is always symmetric
        // NOTE: `ceil` for pooling by default
        if (mGlobalPooling == 1) {
            outputDims.d[1] = 1;
            outputDims.d[2] = 1;
        } else {
            outputDims.d[1] = ceil(h + 2 * mPadH - mKernelH, mStrideH) + 1;
            outputDims.d[2] = ceil(w + 2 * mPadW - mKernelW, mStrideW) + 1;
        }
        return outputDims;
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
#ifdef DEBUG
        float* tmp = (float*)malloc(mH * mW * mChannel * 4);
        cudaMemcpy(tmp, src, mH * mW * mChannel * 4, cudaMemcpyDeviceToHost);
        std::cout << "before ---" << mChannel << std::endl;
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < mH; j++) {
                for (int k = 0; k < mW; k++) {
                    std::cout << *(tmp++) << " ";
                }
                std::cout << std::endl;
            }
        }
#endif
        Pooling(
            dst, src, mChannel, mH, mW, mMethod, mKernelH, mKernelW, mStrideH,
            mStrideW, mPadH, mPadW, stream);
#ifdef DEBUG
        int output_h = (mH - mKernelH) / mStrideH + 1;
        int output_w = (mW - mKernelW) / mStrideW + 1;
        float* tmp2 = (float*)malloc(output_h * output_w * mChannel * 4);
        cudaMemcpy(
            tmp2, dst, output_h * output_w * mChannel * 4,
            cudaMemcpyDeviceToHost);
        std::cout << "after ---" << mChannel << std::endl;
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < output_h; j++) {
                for (int k = 0; k < output_w; k++) {
                    std::cout << *(tmp2++) << " ";
                }
                std::cout << std::endl;
            }
        }
#endif
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 40; }
    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mMethod;
        ((int*)buffer)[1] = mKernelW;
        ((int*)buffer)[2] = mKernelH;
        ((int*)buffer)[3] = mStrideW;
        ((int*)buffer)[4] = mStrideH;
        ((int*)buffer)[5] = mPadW;
        ((int*)buffer)[6] = mPadH;
        ((int*)buffer)[7] = mChannel;
        ((int*)buffer)[8] = mH;
        ((int*)buffer)[9] = mW;
    }
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        auto dims = in[0].dims;
        mChannel = dims.d[0];
        mH = dims.d[1];
        mW = dims.d[2];
        if (mGlobalPooling == 1) {
            mKernelH = mH;
            mKernelW = mW;
        }
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

    const char* getPluginType() const noexcept override { return "POOLING"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new PoolingPlugin(*this);
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

    friend std::ostream& operator<<(std::ostream& os, const PoolingPlugin& c) {
        // clang-format off
        return (os
                << " channel: " << c.mChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " kernel: " << c.mKernelH << " " << c.mKernelW
                << " stride: " << c.mStrideH << " " << c.mStrideW
                << " pad: " << c.mPadH << " " << c.mPadW
                << " method: " << c.mMethod
                << std::endl
        );
        // clang-format on
    }

   private:
    int mChannel;
    int mH;
    int mW;
    int mMethod;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPadH;
    int mPadW;
    int mGlobalPooling;
    std::string mNamespace;
};
