CXX=g++
SDK_PATH=/cuda_sdk
CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_PATH)/C/common/inc/ `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_PATH)/C/lib -lcutil_x86_64 -lcudart `pkg-config --libs opencv`

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(CFLAGS) -c main.cpp -o Debug/main.o
	nvcc $(CUDAFLAGS) -c kernel_gpu.cu -o Debug/kernel_gpu.o
	$(CXX) $(LDFLAGS) Debug/main.o Debug/kernel_gpu.o -o Debug/cuda_skin

clean:
	rm -f Debug/*.o Debug/cuda_skin

