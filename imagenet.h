// 2022-06-24 21:48
#ifndef IMAGENET_H
#define IMAGENET_H

#include <cuda.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_profiler_api.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

static std::array<float, 3> DEFAULT_MEAN = {104.0, 117.0, 123.0};

class ImageNet {
   protected:
    virtual int getWidth() { return 224; }
    virtual int getHeight() { return 224; };
    virtual std::string getImage() { return "data/tench.bmp"; }

    Logger mLogger;

   public:
    ImageNet(
        std::string proto, std::string caffemodel, std::string output,
        float scale = 1.0, std::array<float, 3> mean = DEFAULT_MEAN)
        : mProto(proto),
          mCaffemodel(caffemodel),
          mOutput(output),
          mScale{scale},
          mMean{mean[0], mean[1], mean[2]} {}

    bool build() {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(mLogger));
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(0));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        auto parser = std::unique_ptr<nvcaffeparser1::ICaffeParser>(
            nvcaffeparser1::createCaffeParser());
        constructNetwork(parser, network);

        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(1 << 30);
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

        std::unique_ptr<IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};

        std::unique_ptr<IRuntime> runtime{createInferRuntime(mLogger)};
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));
        mInputDims = network->getInput(0)->getDimensions();
        mOutputDims = network->getOutput(0)->getDimensions();
        return true;
    }

    bool infer() {
        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
            mEngine->createExecutionContext());

        int inputSize = std::accumulate(
            mInputDims.d, mInputDims.d + mInputDims.nbDims, 1,
            std::multiplies<int>());
        int outputSize = std::accumulate(
            mOutputDims.d, mOutputDims.d + mOutputDims.nbDims, 1,
            std::multiplies<int>());
        std::cout << "input size: " << inputSize << std::endl;
        std::cout << "output size: " << outputSize << std::endl;
        void* hostInputBuffer = malloc(inputSize * sizeof(float));
        void* hostOutputBuffer = malloc(outputSize * sizeof(float));
        void* deviceInputBuffer;
        void* deviceOutputBuffer;
        cudaMalloc(&deviceInputBuffer, inputSize * sizeof(float));
        cudaMalloc(&deviceOutputBuffer, outputSize * sizeof(float));

        readBMP(getImage().c_str(), (float*)hostInputBuffer);
        cudaMemcpy(
            deviceInputBuffer, hostInputBuffer, inputSize * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        void* bindings[2] = {
            deviceInputBuffer,
            deviceOutputBuffer,
        };
        cudaProfilerStart();
        context->enqueue(1, bindings, stream, nullptr);
        cudaError_t error = cudaMemcpy(
            hostOutputBuffer, deviceOutputBuffer, outputSize * sizeof(float),
            cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaProfilerStop();
        printf("output:\n");
        for (int i = 0; i < std::min<int>(outputSize, 16); i++) {
            std::cout << ((float*)hostOutputBuffer)[i] << " ";
        }
        std::cout << std::endl;
        for (int i = outputSize - 1; i >= std::max<int>(0, outputSize - 16);
             i--) {
            std::cout << ((float*)hostOutputBuffer)[i] << " ";
        }
        std::cout << std::endl;
        return true;
    }
    bool teardown() {
        nvcaffeparser1::shutdownProtobufLibrary();
        return true;
    }

   private:
    void readBMP(const char* filename, float* data) {
        cv::Mat image;
        image = cv::imread(filename, 1);
        cv::Mat resized_image;
        int WIDTH = getWidth();
        int HEIGHT = getHeight();

        cv::resize(image, resized_image, cv::Size(WIDTH, HEIGHT));
        // NOTE: the caffe model is trained in BGR format
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            data[i] = ((float)resized_image.data[i * 3] - mMean[0]) * mScale;
            data[i + WIDTH * HEIGHT] =
                ((float)resized_image.data[i * 3 + 1] - mMean[1]) * mScale;
            data[i + WIDTH * HEIGHT * 2] =
                ((float)resized_image.data[i * 3 + 2] - mMean[2]) * mScale;
        }
    }
    bool constructNetwork(
        std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network) {
        const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
            parser->parse(
                mProto.c_str(), mCaffemodel.c_str(), *network,
                nvinfer1::DataType::kFLOAT);
        network->markOutput(*blobNameToTensor->find(mOutput.c_str()));
        return true;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob;
    std::string mProto;
    std::string mCaffemodel;
    std::string mOutput;
    float mScale;
    float mMean[3];
};

#endif  // IMAGENET_H
