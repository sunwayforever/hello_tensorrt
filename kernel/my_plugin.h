// 2022-07-11 16:17
#ifndef MY_PLUGIN_H
#define MY_PLUGIN_H
#include <string>

#include "NvInfer.h"
#include "NvInferRuntime.h"

using namespace nvinfer1;
class MyPlugin : public IPluginV2IOExt {
   public:
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }

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
    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

   private:
    std::string mNamespace;
};

#endif  // MY_PLUGIN_H
