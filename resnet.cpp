#include "imagenet.h"
#include "plugin.h"

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;
    ImageNet net(
        "model/resnet18.prototxt", "model/resnet18.caffemodel", "prob");
    net.build();
    net.infer();
    net.teardown();
    return 0;
}
