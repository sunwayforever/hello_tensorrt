# get and patch tensorrt
```
git clone https://github.com/NVIDIA/TensorRT/
pushd TensorRT
git submodule update --init --recursive
git checkout 156c59ae86d454fa89146fe65fa7332dbc8c3c2b
git submodule update
git apply ../tensorrt.diff
popd
```

# build docker image
```
make docker-build
```

# build tensorrt
```
make docker-run
root@docker> cd TensorRT
root@docker> mkdir -p build && cd build
root@docker> cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
root@docker> make
```

# build and run models
```
make docker-run
root@docker> make
root@docker> make run-mnist
root@docker> make run-googlenet
root@docker> make run-mobilenet
root@docker> make run-resnet
```
