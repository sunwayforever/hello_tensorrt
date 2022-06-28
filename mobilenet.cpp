#include "imagenet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;
    ImageNet net(
        "model/mobilenet_v2.prototxt", "model/mobilenet_v2.caffemodel", "prob",
        0.017);
    net.build();
    net.infer();
    net.teardown();
    return 0;
}
