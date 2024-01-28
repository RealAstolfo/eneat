CC = gcc
CXX = g++
LD = ld
AR = ar
AS = as

INC = -I./include -I./vendors/ethreads/include -I./vendors/exstd/include -I./vendors/emath/include
LIB =  -L. -L/usr/lib64 -L/usr/local/lib64

CFLAGS = -march=native -O3 -g -Wall -Wextra -pedantic $(INC)
CXXFLAGS = -std=c++20 $(CFLAGS)
LDFLAGS = $(LIB) -O3

ZLIB = `pkgconf --cflags --libs zlib`

# AI
brain.o:
	${CXX} ${CXXFLAGS} -c src/brain.cpp -o $@

mutation-rate-container.o:
	${CXX} ${CXXFLAGS} -c src/mutation_rate_container.cpp -o $@

pool.o:
	${CXX} ${CXXFLAGS} -c src/pool.cpp -o $@

speciating-parameter-container.o:
	${CXX} ${CXXFLAGS} -c src/speciating_parameter_container.cpp -o $@

model.o:
	${CXX} ${CXXFLAGS} -c src/model.cpp -o $@

ai.o: brain.o mutation-rate-container.o pool.o speciating-parameter-container.o model.o
	${LD} -r $^ -o $@


# Threading
vendors/ethreads/threading.o:
	make -C vendors/ethreads threading.o

#########################################################################################

# NEAT Project implementation
#########################################################################################

neat.o:
	${CXX} ${CXXFLAGS} -c builds/neat.cpp -o $@

neat: ai.o neat.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} $^ -o $@

all: neat

clean:
	-rm -f neat *.o
	make -C vendors/ethreads clean
	make -C vendors/exstd clean
	make -C vendors/emath clean
