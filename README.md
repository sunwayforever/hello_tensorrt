# install cuda, cudnn, tensorrt

1. cuda-11
2. cudnn-8.2.1
3. tensorrt 8.2.1.8

# get and patch tensorrt-oss
```
git clone https://github.com/NVIDIA/TensorRT/
pushd TensorRT
git submodule update --init --recursive
git checkout 156c59ae86d454fa89146fe65fa7332dbc8c3c2b
git submodule update
git apply ../tensorrt.diff
popd
```

# build tensorrt
```
pushd TensorRT
mkdir -p build && cd build
CUDACXX=/opt/anaconda3/envs/cuda-11/bin/nvcc cmake .. -DTRT_LIB_DIR=/opt/anaconda3/envs/cuda-11/lib -DTRT_OUT_DIR=`pwd`/out
make
popd
```

# get pretrianed model

get models from https://mega.nz/folder/X5UD1LDY#8ZI-gAq6AkpcUbE2z4n6RA and save
models to `model` directory

# run

make run-mnist
make run-googlenet
make run-mobilenet
make run-resnet

# run with int8

1.  turn on `CPPFLAGS += -DINT8` in Makefile
2.  make clean
3.  make run-mnist

# run with int8 and calibrator
1.  turn on `CPPFLAGS += -DINT8` and `USE_MNIST_CALIBRATOR=1` in Makefile
2.  make clean
3.  make run-mnist

