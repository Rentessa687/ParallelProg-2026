CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

all: jacobi

jacobi: jacobi.cpp
	$(CXX) $(CXXFLAGS) -o jacobi jacobi.cpp

clean:
	rm -f jacobi output.txt input.txt *.npy

.PHONY: all clean
