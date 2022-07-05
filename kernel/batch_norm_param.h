// 2022-07-05 13:42
#ifndef BATCH_NORM_PARAM_H
#define BATCH_NORM_PARAM_H

// NOTE: POD is required for this class
struct BatchNormParam {
    float mEps;
    float mMovingAverage;
    int mChannel;
    int mH;
    int mW;
};

#endif  // BATCH_NORM_PARAM_H
