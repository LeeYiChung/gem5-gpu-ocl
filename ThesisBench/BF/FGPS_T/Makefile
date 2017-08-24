CXX=g++

all:
	$(CXX) -DGEM5_FUSION -O3 -funroll-loops main.cpp -c -I$(OPENCV)/include -I$(AMDAPP)/include
	$(CXX) *.o -lz -static -static-libgcc -lopencl -L$(OPENCL) `pkg-config --static --cflags --libs opencv` -g -o gem5_fusion_BilateralFilter -L/usr/lib64 -lm5op_x86 -lm -lc -lpthread -L$(M5THREAD)
clean:
	rm -f *.o gem5_fusion_BilateralFilter*
