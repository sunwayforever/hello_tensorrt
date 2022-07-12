#include "segnet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;

    SegNet net("model/enet.prototxt", "model/enet.caffemodel", "pool1_0_4_mask");

    net.build();
    net.infer();
    net.teardown();
    return 0;
}
