CXX      := clang++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra

# Only the SME kernel TU gets the SME flags (never -march=armv9-a+sme2
# on Apple silicon: armv9 promises non-streaming SVE -> SIGILL).
SMEFLAGS := -march=armv8-a+sme2

all: ozksme

ozksme: ozaki-ii.o matmul.o main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp ozaki-ii.hpp matmul.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

ozaki-ii.o: ozaki-ii.cpp ozaki-ii.hpp matmul.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

matmul.o: matmul.cpp matmul.hpp
	$(CXX) $(CXXFLAGS) $(SMEFLAGS) -c -o $@ $<

run: test_scale main
	./test_scale
	./main

clean:
	rm -f ozksme *.o c_ref.bin c_ozaki.bin

.PHONY: run clean
