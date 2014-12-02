CXX=g++
SDK_PATH= $(HOME)/NVIDIA_GPU_Computing_SDK
CUDA_INSTALL_PATH=/usr/local/cuda

CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_PATH)/C/common/inc/ `pkg-config --cflags opencv` -D__STDC_CONSTANT_MACROS `sdl-config --cflags`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_PATH)/C/lib -lcutil_x86_64 -lcudart `sdl-config --libs` `pkg-config --libs opencv` -lavformat -lavcodec -lz -lavutil -lm -lswscale     

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(CFLAGS) -g -c main.cpp -o Debug/main.o  
	nvcc $(CUDAFLAGS) -g -c kernel_gpu.cu -o Debug/kernel_gpu.o
	$(CXX) Debug/main.o Debug/kernel_gpu.o -o Debug/cuda_skin $(LDFLAGS)    

clean:
	rm -f Debug/*.o Debug/cuda_skin

