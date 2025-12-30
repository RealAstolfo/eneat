CXX = zig c++
AR = zig ar

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
	${CXX} ${CXXFLAGS} -c $^ -o $@


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

# NEAT with coroutine-based async runtime (CORO_MAIN pattern)
neat_coro.o:
	${CXX} ${CXXFLAGS} -c builds/neat_coro.cpp -o $@

neat_coro: ai.o neat_coro.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} $^ -o $@

all: neat neat_coro

clean:
	-rm -f neat neat_coro *.o
	make -C vendors/ethreads clean
	make -C vendors/exstd clean
	make -C vendors/emath clean
