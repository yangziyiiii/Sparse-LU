GCC=g++
OPT=-ltapa -lfrt -lglog -lgflags -lOpenCL -I${XILINX_HLS}/include

lu: lu.cpp lu_host.cpp
	$(GCC) -o $@ -O2 $^ $(OPT)