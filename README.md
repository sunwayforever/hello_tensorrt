# get and patch tensorrt

```
git clone https://github.com/NVIDIA/TensorRT/
pushd TensorRT
git submodule update --init --recursive
git checkout 156c59ae86d454fa89146fe65fa7332dbc8c3c2b 
git apply ../tensorrt.diff
popd
```

# build and run

```
make docker-build
make docker-run

root@docker> make
root@docker> make run-mnist
root@docker> make run-googlenet
```
