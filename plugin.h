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
#include "kernel/prior_box_plugin.h"
#include "kernel/relu_plugin.h"
#include "kernel/scale_plugin.h"
#include "kernel/softmax_plugin.h"

#define PLUGIN_LIST(ITEM)              \
    ITEM(Eltwise, ELTWISE);            \
    ITEM(BatchNorm, BATCH_NORM);       \
    ITEM(Convolution, CONVOLUTION);    \
    ITEM(InnerProduct, INNER_PRODUCT); \
    ITEM(LRN, LRN);                    \
    ITEM(Pooling, POOLING);            \
    ITEM(Power, POWER);                \
    ITEM(Relu, RELU);                  \
    ITEM(Scale, SCALE);                \
    ITEM(Softmax, SOFTMAX);            \
    ITEM(Normalize2, NORMALIZE);        \
    ITEM(PriorBox2, PRIOR_BOX);

#define REGISTER_PLUGIN(plugin_type, plugin_name) \
    REGISTER_TENSORRT_PLUGIN(plugin_type##PluginCreator);

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

PLUGIN_LIST(DECLARE_PLUGIN_CREATOR);

#define REGISTER_ALL_PLUGINS PLUGIN_LIST(REGISTER_PLUGIN);

#endif  // PLUGIN_H
