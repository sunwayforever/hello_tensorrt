#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include <stdio.h>
#include <vector>
#include "common.h"


class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class MNISTBatchStream : public IBatchStream
{
public:
    MNISTBatchStream(int batchSize, int maxBatches, const std::string& dataFile, const std::string& labelsFile,
        const std::vector<std::string>& directories)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{3, {1, 28, 28}} //!< We already know the dimensions of MNIST images.
    {
        readDataFile(locateFile(dataFile, directories));
        readLabelsFile(locateFile(labelsFile, directories));
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
        return mData.data() + (mBatchCount * mBatchSize * volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}};
    }

private:
    void readDataFile(const std::string& dataFilePath)
    {
        std::ifstream file{dataFilePath.c_str(), std::ios::binary};

        int magicNumber, numImages, imageH, imageW;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = swapEndianness(magicNumber);
        ASSERT(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

        // Read number of images and dimensions
        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
        file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

        numImages = swapEndianness(numImages);
        imageH = swapEndianness(imageH);
        imageW = swapEndianness(imageW);

        // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
        int numElements = numImages * imageH * imageW;
        std::vector<uint8_t> rawData(numElements);
        file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
        mData.resize(numElements);
        std::transform(
            rawData.begin(), rawData.end(), mData.begin(), [](uint8_t val) { return static_cast<float>(val) / 255.f; });
    }

    void readLabelsFile(const std::string& labelsFilePath)
    {
        std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
        int magicNumber, numImages;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = swapEndianness(magicNumber);
        ASSERT(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        numImages = swapEndianness(numImages);

        std::vector<uint8_t> rawLabels(numImages);
        file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
        mLabels.resize(numImages);
        std::transform(
            rawLabels.begin(), rawLabels.end(), mLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    }

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};

#endif
