// 2022-06-27 18:32
#ifndef PLUGIN_H
#define PLUGIN_H

#include "kernel/batch_norm_plugin.h"
#include "kernel/convolution_plugin.h"
#include "kernel/eltwise_plugin.h"
#include "kernel/inner_product_plugin.h"
#include "kernel/lrn_plugin.h"
#include "kernel/normalize_plugin.h"
#include "kernel/pooling_plugin.h"
#include "kernel/power_plugin.h"
#include "kernel/relu_plugin.h"
#include "kernel/scale_plugin.h"
#include "kernel/softmax_plugin.h"

#define REGISTER_ALL_PLUGINS                                 \
    do {                                                     \
        REGISTER_TENSORRT_PLUGIN(SoftmaxPluginCreator);      \
        REGISTER_TENSORRT_PLUGIN(PowerPluginCreator);        \
        REGISTER_TENSORRT_PLUGIN(ReluPluginCreator);         \
        REGISTER_TENSORRT_PLUGIN(PoolingPluginCreator);      \
        REGISTER_TENSORRT_PLUGIN(InnerProductPluginCreator); \
        REGISTER_TENSORRT_PLUGIN(ConvolutionPluginCreator);  \
        REGISTER_TENSORRT_PLUGIN(LRNPluginCreator);          \
        REGISTER_TENSORRT_PLUGIN(BatchNormPluginCreator);    \
        REGISTER_TENSORRT_PLUGIN(ScalePluginCreator);        \
        REGISTER_TENSORRT_PLUGIN(EltwisePluginCreator);      \
    } while (0)

#define DECLARE_PLUGIN_CREATOR(plugin_type, plugin_name)                       \
    class plugin_type##PluginCreator : public IPluginCreator {                 \
       public:                                                                 \
        const char* getPluginName() const noexcept override {                  \
            return #plugin_name;                                               \
        }                                                                      \
        const char* getPluginVersion() const noexcept override { return "1"; } \
        const PluginFieldCollection* getFieldNames() noexcept override {       \
            return &mFieldCollection;                                          \
        }                                                                      \
        IPluginV2* createPlugin(                                               \
            const char* name,                                                  \
            const PluginFieldCollection* fc) noexcept override {               \
            auto* plugin = new plugin_type##Plugin(*fc);                       \
            mFieldCollection = *fc;                                            \
            mPluginName = name;                                                \
            return plugin;                                                     \
        }                                                                      \
        IPluginV2* deserializePlugin(                                          \
            const char* name, const void* serialData,                          \
            size_t serialLength) noexcept override {                           \
            auto* plugin = new plugin_type##Plugin(serialData, serialLength);  \
            mPluginName = name;                                                \
            return plugin;                                                     \
        }                                                                      \
        void setPluginNamespace(const char* libNamespace) noexcept override {  \
            mNamespace = libNamespace;                                         \
        }                                                                      \
        const char* getPluginNamespace() const noexcept override {             \
            return mNamespace.c_str();                                         \
        }                                                                      \
                                                                               \
       private:                                                                \
        std::string mNamespace;                                                \
        std::string mPluginName;                                               \
        PluginFieldCollection mFieldCollection{0, nullptr};                    \
    };

DECLARE_PLUGIN_CREATOR(Eltwise, ELTWISE);
DECLARE_PLUGIN_CREATOR(BatchNorm, BATCH_NORM);
DECLARE_PLUGIN_CREATOR(Convolution, CONVOLUTION);
DECLARE_PLUGIN_CREATOR(InnerProduct, INNER_PRODUCT);
DECLARE_PLUGIN_CREATOR(LRN, LRN);
DECLARE_PLUGIN_CREATOR(Normalize, NORMALIZE);
DECLARE_PLUGIN_CREATOR(Pooling, POOLING);
DECLARE_PLUGIN_CREATOR(Power, POWER);
DECLARE_PLUGIN_CREATOR(Relu, RELU);
DECLARE_PLUGIN_CREATOR(Scale, SCALE);
DECLARE_PLUGIN_CREATOR(Softmax, SOFTMAX);

#endif  // PLUGIN_H
