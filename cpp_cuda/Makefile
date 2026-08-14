NVCC ?= nvcc
NVCC_FLAGS ?= -O3 -std=c++17 -arch=native --expt-extended-lambda -Xcompiler -Wall -Xcompiler -fPIC

TARGET1 = kronecker_cuda
SRCS1 = kronecker_generator.cu main.cu
OBJS1 = $(SRCS1:.cu=.o)

TARGET2 = bfs_beamer
SRCS2 = bfs_beamer.cu main_bfs.cu
OBJS2 = $(SRCS2:.cu=.o)

TARGET3 = sssp_delta
SRCS3 = sssp_delta.cu main_sssp.cu
OBJS3 = $(SRCS3:.cu=.o)

ALL_TARGETS = $(TARGET1) $(TARGET2) $(TARGET3)

all: $(ALL_TARGETS)

$(TARGET1): $(OBJS1)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS1)

$(TARGET2): $(OBJS2)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS2)

$(TARGET3): $(OBJS3)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS3)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o $(ALL_TARGETS)

.PHONY: all clean
