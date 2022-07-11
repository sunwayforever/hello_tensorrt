// 2022-07-11 19:51
#ifndef POOLING_PARAM_H
#define POOLING_PARAM_H

struct PoolingParam {
    int mChannel;
    int mH;
    int mW;
    int mMethod;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPaddingH;
    int mPaddingW;
    int mGlobalPooling;
    int mNeedMask;
};

#endif  // POOLING_PARAM_H
