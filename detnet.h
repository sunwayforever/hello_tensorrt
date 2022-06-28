// 2022-06-24 21:48
#ifndef DETNET_H
#define DETNET_H
#include "imagenet.h"

using namespace nvinfer1;

class DetNet : public ImageNet {
   protected:
    int getWidth() { return 300; };
    int getHeight() { return 300; };

    std::string getImage() { return "data/dog.bmp"; };

   public:
    DetNet(std::string proto, std::string caffemodel, std::string output)
        : ImageNet(proto, caffemodel, output) {}

    bool build() {
        initLibNvInferPlugins(&mLogger, "");
        return ImageNet::build();
    }
};

#endif  // DETNET_H
