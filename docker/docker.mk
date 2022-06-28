CPPFLAGS := -I/usr/include/opencv4 -I/usr/local/cuda-11/include
# CPPFLAGS += -DINT8
CXXFLAGS := -g -O0 -MMD
LDFLAGS :=  -L/workspace/TensorRT/build/out -L/usr/local/cuda-11/lib64
LDLIBS := -lnvcaffeparser -lnvinfer -lnvinfer_plugin -lcudnn -lcudart -lstdc++ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

SRC := $(wildcard *.cpp)
OBJ := $(patsubst %.cpp,%.o,${SRC})
APP := $(patsubst %.cpp,%.elf,${SRC})
RUN_APP := $(patsubst %.cpp,run-%,${SRC})

all: ${APP}

DEP := $(OBJ:.o=.d)
-include ${DEP}

CUDA_KERNEL_SRC:=$(wildcard kernel/*.cu)
CUDA_OBJ := $(patsubst %.cu,%.o,${CUDA_KERNEL_SRC})

%.o:%.cu
	nvcc -c $^ -o $@

.PRECIOUS: ${CUDA_OBJ}

%.elf:%.o ${CUDA_OBJ}
	gcc $^ ${LDFLAGS} ${LDLIBS} -o $@

${RUN_APP}:run-%:%.elf
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/TensorRT/build/out:/usr/local/cuda-11/lib64" ./$<

clean:
	rm ${OBJ} ${APP} ${DEP} ${CUDA_OBJ}

docker-build:
	docker build -t hello_tensorrt .

docker-run:
	docker run --net=host --gpus all --rm -v ${PWD}:/workspace/ -it hello_tensorrt
