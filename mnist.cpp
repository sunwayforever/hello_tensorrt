#include <cuda_runtime_api.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "kernel/convolution_plugin.h"
#include "kernel/inner_product_plugin.h"
#include "kernel/pooling_plugin.h"
#include "kernel/power_plugin.h"
#include "kernel/relu_plugin.h"
#include "kernel/softmax_plugin.h"

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

class SampleMNIST {
   public:
    SampleMNIST() {}
    bool build();
    bool infer();
    bool teardown();

   private:
    bool constructNetwork(
        std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob;
};

Logger logger;

bool SampleMNIST::build() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvcaffeparser1::ICaffeParser>(
        nvcaffeparser1::createCaffeParser());
    constructNetwork(parser, network);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
#ifdef INT8
    config->setFlag(BuilderFlag::kINT8);
    config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
#endif

    std::unique_ptr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};

    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();
    return true;
}

#ifdef INT8
inline void setAllDynamicRanges(
    INetworkDefinition* network, float inRange = 2.0f, float outRange = 4.0f) {
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++) {
            ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet()) {
                input->setDynamicRange(-inRange, inRange);
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
            ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet()) {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING) {
                    output->setDynamicRange(-inRange, inRange);
                } else {
                    output->setDynamicRange(-outRange, outRange);
                }
            }
        }
    }
}
#endif

bool SampleMNIST::constructNetwork(
    std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network) {
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        "model/mnist.prototxt", "model/mnist.caffemodel", *network,
        nvinfer1::DataType::kFLOAT);

    network->markOutput(*blobNameToTensor->find("prob"));

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    mMeanBlob = std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob>(
        parser->parseBinaryProto("model/mnist_mean.binaryproto"));
    nvinfer1::Weights meanWeights{
        nvinfer1::DataType::kFLOAT, mMeanBlob->getData(),
        inputDims.d[1] * inputDims.d[2]};

    float maxMean = 255.0;

    auto mean = network->addConstant(
        nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    if (!mean->getOutput(0)->setDynamicRange(-maxMean, maxMean)) {
        return false;
    }
    if (!network->getInput(0)->setDynamicRange(-maxMean, maxMean)) {
        return false;
    }
    auto meanSub = network->addElementWise(
        *network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    if (!meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean)) {
        return false;
    }
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
#ifdef INT8
    setAllDynamicRanges(network.get(), 127.0f, 127.0f);

    // int8 for conv
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        if (std::string(layer->getName()).find("conv1") == 0) {
            layer->setPrecision(DataType::kINT8);
        }
    }
#endif
    return true;
}

bool SampleMNIST::teardown() {
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

inline void readImage(
    const std::string& fileName, uint8_t* buffer, int inH, int inW) {
    std::ifstream infile(fileName, std::ifstream::binary);
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

bool SampleMNIST::infer() {
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());

    int inputSize = std::accumulate(
        mInputDims.d, mInputDims.d + mInputDims.nbDims, 1,
        std::multiplies<int>());
    int outputSize = std::accumulate(
        mOutputDims.d, mOutputDims.d + mOutputDims.nbDims, 1,
        std::multiplies<int>());

    void* hostInputBuffer = malloc(inputSize * sizeof(float));
    void* hostOutputBuffer = malloc(outputSize * sizeof(float));
    void* deviceInputBuffer;
    void* deviceOutputBuffer;
    cudaMalloc(&deviceInputBuffer, inputSize * sizeof(float));
    cudaMalloc(&deviceOutputBuffer, outputSize * sizeof(float));

    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    std::vector<uint8_t> imageData(inputH * inputW);
    readImage("data/0.pgm", imageData.data(), inputH, inputW);

    for (int i = 0; i < inputH * inputW; i++) {
        ((float*)hostInputBuffer)[i] = float(imageData[i]);
    }
    cudaMemcpy(
        deviceInputBuffer, hostInputBuffer, inputSize * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* bindings[2] = {
        deviceInputBuffer,
        deviceOutputBuffer,
    };
    context->enqueue(1, bindings, stream, nullptr);
    cudaError_t error = cudaMemcpy(
        hostOutputBuffer, deviceOutputBuffer, outputSize * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    printf("output:\n");
    for (int i = 0; i < outputSize; i++) {
        std::cout << ((float*)hostOutputBuffer)[i] << " ";
    }
    std::cout << std::endl;
    return true;
}

int main(int argc, char** argv) {
#ifdef INT8
    REGISTER_TENSORRT_PLUGIN(ConvolutionPluginCreator);
#else
    REGISTER_TENSORRT_PLUGIN(SoftmaxPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PowerPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ReluPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PoolingPluginCreator);
    REGISTER_TENSORRT_PLUGIN(InnerProductPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ConvolutionPluginCreator);
#endif
    SampleMNIST sample;
    sample.build();
    sample.infer();
    sample.teardown();
    return 0;
}
