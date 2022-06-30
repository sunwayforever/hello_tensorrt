#include "detnet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;
    DetNet net(
        "model/ssd.prototxt", "model/ssd.caffemodel",
        "conv4_3_norm_mbox_priorbox");

    net.build();
    net.infer();
    net.teardown();
    return 0;
}
