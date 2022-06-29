#include "detnet.h"
#include "plugin.h"

#include "kernel/normalize_plugin.h"

int main(int argc, char** argv) {
    // REGISTER_ALL_PLUGINS;
    REGISTER_TENSORRT_PLUGIN(NormalizePluginCreator);
    DetNet net("model/ssd.prototxt", "model/ssd.caffemodel", "conv4_3_norm");
    net.build();
    net.infer();
    net.teardown();
    return 0;
}
