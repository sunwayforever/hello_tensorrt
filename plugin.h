// 2022-06-27 18:32
#ifndef PLUGIN_H
#define PLUGIN_H

#include "kernel/batch_norm_plugin.h"
#include "kernel/convolution_plugin.h"
#include "kernel/eltwise_plugin.h"
#include "kernel/inner_product_plugin.h"
#include "kernel/lrn_plugin.h"
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

#endif  // PLUGIN_H
