CC=g++

CXXFLAGS += -std=c++11 
CXXFLAGS += -fopenmp -g -O3 -Wall -static -I. -msse2
LDLIBS += -lboost_program_options  -lpthread -lm

.PHONY:	all clean

PROGS= nndes fvecsnndes

all:	$(PROGS)

clean:
	rm -f $(PROGS)

