# install cuda, cudnn, tensorrt

# patch tensorrt

1.  git clone https://github.com/NVIDIA/TensorRT/, checkout to
    156c59ae86d454fa89146fe65fa7332dbc8c3c2b and apply `tensorrt.diff`

2.  build TensorRT

3.  change Makefile based on your local config

# run

make run\_mnist

make run\_googlenet

# run with int8

1.  turn on `CPPFLAGS += -DINT8` in Makefile
2.  make clean
3.  make run\_mnist

# TODO

1. ssd: PriorBox
2. enet: deconvolution, upsample and dilated convolution