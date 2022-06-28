# install cuda, cudnn, tensorrt

1. cuda-11
2. cudnn-8.2.1
3. tensorrt 8.2.1.8

# patch tensorrt-oss

1.  get https://github.com/NVIDIA/TensorRT/, checkout to
    156c59ae86d454fa89146fe65fa7332dbc8c3c2b and apply `tensorrt.diff`

2.  build TensorRT

3.  change Makefile based on your local config

# run

make run-mnist
make run-googlenet
make run-mobilenet
make run-resnet

# run with int8

1.  turn on `CPPFLAGS += -DINT8` in Makefile
2.  make clean
3.  make run-mnist
