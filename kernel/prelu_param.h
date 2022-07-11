// 2022-07-11 18:02
#ifndef PRELU_PARAM_H
#define PRELU_PARAM_H

struct PReLUParam {
    int mChannelShared;
    int mChannel;
    int mTotalSize;
    int mSlopeWeightsCount;
};

#endif  // PRELU_PARAM_H
