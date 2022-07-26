// 2022-07-07 19:55
#ifndef CONVOLUTION_PARAM_H
#define CONVOLUTION_PARAM_H

struct ConvolutionParam {
    int mOutputChannel;
    int mInputChannel;
    int mGroup;
    int mH;
    int mW;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPaddingH;
    int mPaddingW;
    int mType;
    float mInputScale;
    float mOutputScale;
    float mKernelScale;
    int mDilationH;
    int mDilationW;
    int mKernelWeightsSize;
    int mBiasWeightsSize;
    // for deconv
    int mOrigH;
    int mOrigW;
    int mOrigStrideH;
    int mOrigStrideW;
};

#endif  // CONVOLUTION_PARAM_H
