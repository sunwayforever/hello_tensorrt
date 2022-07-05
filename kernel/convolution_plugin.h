// 2022-06-14 10:53
#include <assert.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <cmath>

extern void Convolution(
    float* dst, const float* src, int input_channel, int output_channel,
    int group, int h, int w, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w,
    float* kernel, float* bias, void* workspace, cudaStream_t);

extern void ConvolutionInt8(
    int8_t* dst, const int8_t* src, float input_scale, float output_scale,float kernel_scale,
    int input_channel, int output_channel, int group, int h, int w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h,
    int padding_w, int8_t* kernel, int8_t* bias, cudaStream_t stream);

using namespace nvinfer1;

class ConvolutionPlugin : public IPluginV2IOExt {
   public:
    ConvolutionPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "num_output") {
                this->mOutputChannel = *((int*)field.data);
            }
            if (std::string(field.name) == "group") {
                this->mGroup = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_weights") {
                this->mKernelWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "bias_weights") {
                this->mBiasWeights = *(Weights*)field.data;
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
            if (std::string(field.name) == "dilation_h") {
                this->mDilationH = *((int*)field.data);
            }
            if (std::string(field.name) == "dilation_w") {
                this->mDilationW = *((int*)field.data);
            }
        }
    }

    ConvolutionPlugin(const void* data, size_t length) {
        mInputChannel = ((int*)data)[0];
        mOutputChannel = ((int*)data)[1];
        mGroup = ((int*)data)[2];
        mH = ((int*)data)[3];
        mW = ((int*)data)[4];
        mKernelH = ((int*)data)[5];
        mKernelW = ((int*)data)[6];
        mStrideH = ((int*)data)[7];
        mStrideW = ((int*)data)[8];
        mPadH = ((int*)data)[9];
        mPadW = ((int*)data)[10];

        int kc = ((int*)data)[11];
        int bc = ((int*)data)[12];
        mType = ((int*)data)[13];
        //mInputScale = ((int*)data)[14];
        //mOutputScale = ((int*)data)[15];
        mInputScale = ((float*)data)[14]; //int-float-4byte
        mOutputScale =((float*)data)[15]; //int-float-4byte
        mDilationH = ((int*)data)[16];
        mDilationW = ((int*)data)[17];
        float* kernel = (float*)malloc(kc * 4);
        float* bias = (float*)malloc(bc * 4);
        memcpy(kernel, ((int*)data) + 18, kc * 4);
        memcpy(bias, ((int*)data) + 18 + kc, bc * 4);
        mKernelWeights = Weights{
            .type = DataType::kFLOAT,
            .values = kernel,
            .count = kc,
        };
        mBiasWeights = Weights{
            .type = DataType::kFLOAT,
            .values = bias,
            .count = bc,
        };
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    static int floor(int a, int b) {
        assert(a >= 0);
        assert(b > 0);
        return a / b;
    }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = mOutputChannel;
        // NOTE: `floor` for convolution
        outputDims.d[1] =
            floor(h + 2 * mPadH - (mDilationH * (mKernelH - 1) + 1), mStrideH) +
            1;
        outputDims.d[2] =
            floor(w + 2 * mPadW - (mDilationW * (mKernelW - 1) + 1), mStrideW) +
            1;

        return outputDims;
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return (mKernelWeights.count + mBiasWeights.count) * 4;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        if (mType == (int)DataType::kFLOAT) {
            float* dst = reinterpret_cast<float*>(outputs[0]);
            const float* src = reinterpret_cast<const float*>(inputs[0]);
            Convolution(
                dst, src, mInputChannel, mOutputChannel, mGroup, mH, mW,
                mKernelH, mKernelW, mStrideH, mStrideW, mPadH, mPadW,
                mDilationH, mDilationW, (float*)mKernelWeights.values,
                mBiasWeights.count == 0 ? NULL : (float*)mBiasWeights.values,
                workspace, stream);
        } else {
            int8_t* dst = reinterpret_cast<int8_t*>(outputs[0]);
            const int8_t* src = reinterpret_cast<const int8_t*>(inputs[0]);
            /*ConvolutionInt8(
                dst, src, mInputScale, mOutputScale, mInputChannel,
                mOutputChannel, mGroup, mH, mW, mKernelH, mKernelW, mStrideH,
                mStrideW, mPadH, mPadW, (float*)mKernelWeights.values,
                mBiasWeights.count == 0 ? NULL : (float*)mBiasWeights.values,
                stream);*/
            ConvolutionInt8(
                dst, src, mInputScale, mOutputScale, mKernelScale,
                mInputChannel, mOutputChannel, mGroup, mH, mW,
                mKernelH, mKernelW, mStrideH, mStrideW, mPadH,
                mPadW, mKernelWeights_I8, mBiasWeights.count == 0 ? NULL : mBiasWeights_I8, stream);
        }

        return 0;
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        mType = (int)in[0].type;
        mInputScale = in[0].scale;
        mOutputScale = out[0].scale;
        auto dims = in[0].dims;
        mInputChannel = dims.d[0];
        mH = dims.d[1];
        mW = dims.d[2];

        //
        mBiasWeights_I8 = (int8_t*)malloc(sizeof(int8_t) * mOutputChannel);
        mKernelWeights_I8 = (int8_t*)malloc(sizeof(int8_t)*(mInputChannel * mOutputChannel * mKernelH * mKernelW));
        float kernel_max = ((float*)mKernelWeights.values)[0];
        float kernel_min = ((float*)mKernelWeights.values)[0];
        for(int i=0; i<mInputChannel * mOutputChannel * mKernelH * mKernelW; i++){
            if (((float*)mKernelWeights.values)[i] > kernel_max){
                kernel_max = ((float*)mKernelWeights.values)[i];
            }
            if (((float*)mKernelWeights.values)[i] < kernel_min){
                kernel_min = ((float*)mKernelWeights.values)[i];
            }
        }

        mKernelScale = (float)std::max(std::fabs(kernel_max),std::fabs(kernel_min))/127;

        for (int i=0; i<mInputChannel * mOutputChannel * mKernelH * mKernelW; i++){
            mKernelWeights_I8[i] = (int8_t)(((float*)(mKernelWeights.values))[i]/mKernelScale); //Q
        }

        if (mBiasWeights.count != 0){
            for (int i = 0; i < mOutputChannel; i++){
                mBiasWeights_I8[i] = (int8_t)(((float*)(mBiasWeights.values))[i]/(mKernelScale * mInputScale));//Q
            }
        }
        //
    }

    size_t getSerializationSize() const noexcept override {
        return (18 + mKernelWeights.count + mBiasWeights.count) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mInputChannel;
        ((int*)buffer)[1] = mOutputChannel;
        ((int*)buffer)[2] = mGroup;
        ((int*)buffer)[3] = mH;
        ((int*)buffer)[4] = mW;
        ((int*)buffer)[5] = mKernelH;
        ((int*)buffer)[6] = mKernelW;
        ((int*)buffer)[7] = mStrideH;
        ((int*)buffer)[8] = mStrideW;
        ((int*)buffer)[9] = mPadH;
        ((int*)buffer)[10] = mPadW;
        ((int*)buffer)[11] = mKernelWeights.count;
        ((int*)buffer)[12] = mBiasWeights.count;
        ((int*)buffer)[13] = mType;
        //((int*)buffer)[14] = mInputScale;
        //((int*)buffer)[15] = mOutputScale;
        ((float*)buffer)[14] = mInputScale; //int-float-4byte
        ((float*)buffer)[15] = mOutputScale; //int-float-4byte
        ((int*)buffer)[16] = mDilationH;
        ((int*)buffer)[17] = mDilationW;
        memcpy(
            ((int*)buffer) + 18, mKernelWeights.values,
            mKernelWeights.count * 4);
        memcpy(
            ((int*)buffer) + 18 + mKernelWeights.count, mBiasWeights.values,
            mBiasWeights.count * 4);
    }


    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        // std::cout << (int)inOut[pos].format << " " << (int)inOut[pos].type
        //           << " " << (int)inOut[0].type << std::endl;
        return inOut[pos].format == TensorFormat::kLINEAR &&
               (inOut[pos].type == DataType::kFLOAT ||
                inOut[pos].type == DataType::kINT8) &&
               inOut[pos].type == inOut[0].type;
    }
    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override {
        return "CONVOLUTION";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new ConvolutionPlugin(*this);
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
        std::ostream& os, const ConvolutionPlugin& c) {
        // clang-format off
        return (os
                << " input channel: " << c.mInputChannel
                << " output channel: " << c.mOutputChannel
                << " group: " << c.mGroup
                << " h: " << c.mH
                << " w: " << c.mW
                << " kernel: " << c.mKernelH << " " << c.mKernelW
                << " stride: " << c.mStrideH << " " << c.mStrideW
                << " pad: " << c.mPadH << " " << c.mPadW
                << " type: " << c.mType << " scale: " << c.mInputScale << " " << c.mOutputScale
                << " dilation: " << c.mDilationH << " " << c.mDilationW
                << " kernel weights: " << c.mKernelWeights.count
                << " bias weights: " << c.mBiasWeights.count
                << std::endl
        );
        // clang-format on
    }

   private:
    int mOutputChannel;
    int mInputChannel;
    int mGroup;
    int mH;
    int mW;
    Weights mKernelWeights;
    Weights mBiasWeights;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPadH;
    int mPadW;
    int mType;
    //int mInputScale;
    //int mOutputScale;
    float mInputScale;
    float mOutputScale;
    float mKernelScale;
    int8_t* mKernelWeights_I8;
    int8_t* mBiasWeights_I8;
    int mDilationH;
    int mDilationW;
    std::string mNamespace;
};
