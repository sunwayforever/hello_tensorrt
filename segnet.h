// 2022-06-24 21:48
#ifndef SEGNET_H
#define SEGNET_H
#include "imagenet.h"

using namespace nvinfer1;

class SegNet : public ImageNet {
   protected:
    int getWidth() { return 1024; };
    int getHeight() { return 512; };

    std::string getImage() { return "data/munich.bmp"; };

   public:
    SegNet(std::string proto, std::string caffemodel, std::string output)
        : ImageNet(
              proto, caffemodel, output, 1.0,
              std::array<float, 3>{0.0, 0.0, 0.0}) {}
};

#endif  // SEGNET_H
