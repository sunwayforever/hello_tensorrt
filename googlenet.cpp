#include "imagenet.h"
#include "plugin.h"
int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;
    ImageNet net(
        "model/googlenet.prototxt", "model/googlenet.caffemodel", "prob");
    net.build();
    net.infer();
    net.teardown();
    return 0;
}
