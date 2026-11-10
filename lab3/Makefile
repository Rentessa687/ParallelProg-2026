MPICXX ?= mpicxx
CXXFLAGS = -O2 -std=c++11 -Wall -Wextra

all: jacobi_mpi

jacobi_mpi: jacobi_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o jacobi_mpi jacobi_mpi.cpp

clean:
	rm -f jacobi_mpi output.txt input.txt *.npy

.PHONY: all clean
