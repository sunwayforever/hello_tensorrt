#include "segnet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;

    SegNet net("model/enet.prototxt", "model/enet.caffemodel", "deconv6_0_0");

    net.build();
    net.infer();
    net.teardown();
    return 0;
}
