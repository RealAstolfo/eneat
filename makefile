CXX = zig c++
AR = zig ar

INC = -I./include -I./vendors/ethreads/include -I./vendors/exstd/include -I./vendors/emath/include
LIB =  -L. -L/usr/lib64 -L/usr/local/lib64

CFLAGS = -march=native -O3 -g -Wall -Wextra -pedantic $(INC)
CXXFLAGS = -std=c++20 $(CFLAGS)
LDFLAGS = $(LIB) -O3

# Debug flags for Valgrind (no optimization for accurate line numbers)
CXX_DEBUG = g++
CFLAGS_DEBUG = -O0 -g3 -Wall -Wextra -pedantic $(INC)
CXXFLAGS_DEBUG = $(CFLAGS_DEBUG) -std=c++20

ZLIB = `pkgconf --cflags --libs zlib`
RAYLIB = `pkg-config --cflags --libs raylib`

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

# Network Visualizer (raylib)
network-visualizer.o:
	${CXX} ${CXXFLAGS} -c src/network_visualizer.cpp -o $@

# NEAT with coroutine-based async runtime (CORO_MAIN pattern) + visualization
neat_coro.o:
	${CXX} ${CXXFLAGS} -c builds/neat_coro.cpp -o $@

neat_coro: ai.o neat_coro.o network-visualizer.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${RAYLIB} $^ -o $@

all: neat neat_coro

#########################################################################################
# Debug builds for Valgrind
#########################################################################################

brain-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/brain.cpp -o $@

mutation-rate-container-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/mutation_rate_container.cpp -o $@

pool-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/pool.cpp -o $@

speciating-parameter-container-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/speciating_parameter_container.cpp -o $@

model-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/model.cpp -o $@

ai-debug.o: brain-debug.o mutation-rate-container-debug.o pool-debug.o \
            speciating-parameter-container-debug.o model-debug.o
	ld -r $^ -o $@

vendors/ethreads/threading-debug.o:
	make -C vendors/ethreads threading-debug.o

neat-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c builds/neat.cpp -o $@

neat-valgrind: ai-debug.o neat-debug.o vendors/ethreads/threading-debug.o
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} ${ZLIB} $^ -o $@

neat_coro-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c builds/neat_coro.cpp -o $@

network-visualizer-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/network_visualizer.cpp -o $@

neat_coro-valgrind: ai-debug.o neat_coro-debug.o network-visualizer-debug.o \
                    vendors/ethreads/threading-debug.o
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} ${ZLIB} ${RAYLIB} $^ -o $@

# Helgrind targets
valgrind-all: neat-valgrind neat_coro-valgrind

helgrind-neat: neat-valgrind
	valgrind --tool=helgrind --suppressions=vendors/ethreads/valgrind.supp ./neat-valgrind

helgrind-coro: neat_coro-valgrind
	valgrind --tool=helgrind --suppressions=vendors/ethreads/valgrind.supp ./neat_coro-valgrind

clean:
	-rm -f neat neat_coro neat-valgrind neat_coro-valgrind *.o *-debug.o
	make -C vendors/ethreads clean
	make -C vendors/exstd clean
	make -C vendors/emath clean
