CPPFLAGS := -I /opt/anaconda3/envs/cuda-11/include/ -I/usr/include/opencv4 -I${PWD}/TensorRT/samples/common
# CPPFLAGS += -DINT8
# USE_MNIST_CALIBRATOR=1
CXXFLAGS := -g -O0 -MMD -Wno-deprecated-declarations
LDFLAGS := -L/opt/anaconda3/envs/cuda-11/lib -L/opt/anaconda3/envs/cuda-11/lib64 -L${PWD}/TensorRT/build/out
LDLIBS := -lnvcaffeparser -lnvinfer -lnvinfer_plugin -lcudnn -lcudart -lstdc++ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

NVCC := /opt/anaconda3/envs/cuda-11/bin/nvcc
SRC := $(wildcard *.cpp)
OBJ := $(patsubst %.cpp,%.o,${SRC})
APP := $(patsubst %.cpp,%.elf,${SRC})
RUN_APP := $(patsubst %.cpp,run-%,${SRC})
TEST_APP := $(patsubst %.cpp,test-%,${SRC})

all: ${APP}

DEP := $(OBJ:.o=.d)
-include ${DEP}

CUDA_KERNEL_SRC:=$(wildcard kernel/*.cu)
CUDA_OBJ := $(patsubst %.cu,%.o,${CUDA_KERNEL_SRC})

ifeq (${USE_MNIST_CALIBRATOR}, 1)
        EXTRA_OBJ := ${PWD}/TensorRT/build/samples/sampleINT8/CMakeFiles/sample_int8.dir/__/common/logger.cpp.o
        CPPFLAGS += -DUSE_MNIST_CALIBRATOR
endif

%.o:%.cu
	${NVCC} -c $^ -o $@

.PRECIOUS: ${CUDA_OBJ} ${OBJ}

%.elf:%.o ${CUDA_OBJ} ${EXTRA_OBJ}
	gcc $^ ${LDFLAGS} ${LDLIBS} -o $@

${RUN_APP}:run-%:%.elf
	LD_LIBRARY_PATH="${PWD}/TensorRT/build/out:/opt/anaconda3/envs/cuda-11/lib64:/opt/anaconda3/envs/cuda-11/lib"  ./$<

${TEST_APP}:test-%:%.elf
	@LD_LIBRARY_PATH="${PWD}/TensorRT/build/out:/opt/anaconda3/envs/cuda-11/lib64:/opt/anaconda3/envs/cuda-11/lib"  python3 ./run-test.py $<

test-all:${TEST_APP}

clean:
	rm -rf ${OBJ} ${APP} ${DEP} ${CUDA_OBJ}

build-tensorrt:
	cd TensorRT; mkdir -p build && cd build; CUDACXX=/opt/anaconda3/envs/cuda-11/bin/nvcc cmake .. -DTRT_LIB_DIR=/opt/anaconda3/envs/cuda-11/lib -DTRT_OUT_DIR=`pwd`/out; make; cd ..

get-tensorrt:
	git clone https://github.com/NVIDIA/TensorRT/; cd TensorRT; git submodule update --init --recursive; git checkout 156c59ae86d454fa89146fe65fa7332dbc8c3c2b; git submodule update; git apply ../tensorrt.diff; cd ..

get-mnist-dataset:
	cd mnist_calibration_data; wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz; wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz; gunzip train-images-idx3-ubyte.gz; gunzip train-labels-idx1-ubyte.gz; cd ..
