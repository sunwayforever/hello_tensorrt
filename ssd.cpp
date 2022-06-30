#include "detnet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    // REGISTER_ALL_PLUGINS;
    REGISTER_TENSORRT_PLUGIN(EltwisePluginCreator);
    REGISTER_TENSORRT_PLUGIN(BatchNormPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ConvolutionPluginCreator);
    REGISTER_TENSORRT_PLUGIN(InnerProductPluginCreator);
    REGISTER_TENSORRT_PLUGIN(LRNPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PoolingPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PowerPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ReluPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ScalePluginCreator);
    // REGISTER_TENSORRT_PLUGIN(SoftmaxPluginCreator);
    REGISTER_TENSORRT_PLUGIN(Normalize2PluginCreator);
    REGISTER_TENSORRT_PLUGIN(PriorBox2PluginCreator);

    DetNet net("model/ssd.prototxt", "model/ssd.caffemodel", "detection_out");

    net.build();
    net.infer();
    net.teardown();
    return 0;
}
