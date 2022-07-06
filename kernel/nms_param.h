// 2022-07-06 14:29
#ifndef NMS_PARAM_H
#define NMS_PARAM_H

struct NMSParam {
    int mNumClasses;
    int mKeepTopK;
    float mConfidenceThreshold;
    int mNMSTopK;
    float mNMSThreshold;
    int mNumBox;
};

#endif  // NMS_PARAM_H
