#include <cuda_runtime_api.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin.h"

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST() {}

    bool build();

    bool infer();

private:
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network,
        std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser);

};

Logger logger;

bool SampleOnnxMNIST::build() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    constructNetwork(builder, network, config, parser);

    std::unique_ptr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};

    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();
    return true;
}

bool SampleOnnxMNIST::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                      std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                      std::unique_ptr<nvonnxparser::IParser>& parser) {
    parser->parseFromFile("model/mnist_sim.onnx",
        static_cast<int>(ILogger::Severity::kWARNING));
    config->setMaxWorkspaceSize(1 << 20);
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

bool SampleOnnxMNIST::infer()
{
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

    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<uint8_t> imageData(inputH * inputW);
    readImage("data/0.pgm", imageData.data(), inputH, inputW);

    for (int i = 0; i < inputH * inputW; i++) {
        ((float*)hostInputBuffer)[i] = float(imageData[i]);//1.0 - float(imageData[i]/255.0);
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
    context->enqueueV2( bindings, stream, nullptr);
    cudaError_t error = cudaMemcpy(
        hostOutputBuffer, deviceOutputBuffer, outputSize * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    printf("output:\n");
    for (int i = 0; i < std::min<int>(outputSize, 16); i++) {
        std::cout << ((float*)hostOutputBuffer)[i] << " ";
    }
    std::cout << std::endl;
    for (int i = outputSize - 1; i >= std::max<int>(0, outputSize - 16); i--) {
        std::cout << ((float*)hostOutputBuffer)[i] << " ";
    }
    std::cout << std::endl;
    return true;

}

int main(int argc, char** argv) {
    REGISTER_ALL_PLUGINS;

    SampleOnnxMNIST sample;
    sample.build();
    sample.infer();
    
    return 0;
}