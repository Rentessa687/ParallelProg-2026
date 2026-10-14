CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

UNAME := $(shell uname -s)

ifeq ($(UNAME),Darwin)
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
OMP_FLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
OMP_LIBS = -L$(LIBOMP_PREFIX)/lib -lomp
else
OMP_FLAGS = -fopenmp
OMP_LIBS =
endif

all: jacobi_omp jacobi_seq

jacobi_omp: jacobi.cpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o jacobi_omp jacobi.cpp $(OMP_LIBS)

jacobi_seq: jacobi.cpp
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -o jacobi_seq jacobi.cpp

clean:
	rm -f jacobi_omp jacobi_seq output.txt input.txt *.npy

.PHONY: all clean
