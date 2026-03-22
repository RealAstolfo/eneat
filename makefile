CXX = zig c++
AR = zig ar

INC = -I./include -I./vendors/ethreads/include -I./vendors/exstd/include -I./vendors/emath/include
LIB =  -L. -L/usr/lib64 -L/usr/local/lib64

CFLAGS = -march=native -O3 -g -Wall -Wextra -pedantic $(INC)
CXXFLAGS = -std=c++20 $(CFLAGS)
LDFLAGS = $(LIB) -O3

# Debug flags for Valgrind (no optimization for accurate line numbers)
CXX_DEBUG = g++
CFLAGS_DEBUG = -O0 -g3 -Wall -Wextra -pedantic $(INC) $(ZLIB_CFLAGS)
CXXFLAGS_DEBUG = $(CFLAGS_DEBUG) -std=c++20

ZLIB_CFLAGS = `pkgconf --cflags zlib`
ZLIB_LIBS = `pkgconf --libs zlib`
ZLIB = $(ZLIB_CFLAGS) $(ZLIB_LIBS)
RAYLIB_CFLAGS = `pkg-config --cflags raylib`
RAYLIB_LIBS = `pkg-config --libs raylib`
RAYLIB = $(RAYLIB_CFLAGS) $(RAYLIB_LIBS)
ETHREADS_LIBS = -lmimalloc -luring

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
	${CXX} ${CXXFLAGS} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

# Network Visualizer (raylib)
network-visualizer.o:
	${CXX} ${CXXFLAGS} -c src/network_visualizer.cpp -o $@

# NEAT with coroutine-based async runtime (CORO_MAIN pattern) + visualization
neat_coro.o:
	${CXX} ${CXXFLAGS} -c builds/neat_coro.cpp -o $@

neat_coro: ai.o neat_coro.o network-visualizer.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${RAYLIB} ${ETHREADS_LIBS} $^ -o $@

# Genome control demo (gene expression, crossover variants)
neat_genome_control.o:
	${CXX} ${CXXFLAGS} -c builds/neat_genome_control.cpp -o $@

neat_genome_control: ai.o neat_genome_control.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

# Hebbian learning demo (T-maze, temporal memory)
neat_hebbian.o:
	${CXX} ${CXXFLAGS} -c builds/neat_hebbian.cpp -o $@

neat_hebbian: ai.o neat_hebbian.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

# Coevolution demo (predator-prey, rtNEAT control)
neat_coevolution.o:
	${CXX} ${CXXFLAGS} -c builds/neat_coevolution.cpp -o $@

neat_coevolution: ai.o neat_coevolution.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

# Interactive visualization demo (custom labels, output override)
neat_visualized.o:
	${CXX} ${CXXFLAGS} -c builds/neat_visualized.cpp -o $@

neat_visualized: ai.o neat_visualized.o network-visualizer.o vendors/ethreads/threading.o
	${CXX} ${CXXFLAGS} ${ZLIB} ${RAYLIB} ${ETHREADS_LIBS} $^ -o $@

all: neat neat_coro neat_genome_control neat_hebbian neat_coevolution neat_visualized

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
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

neat_coro-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c builds/neat_coro.cpp -o $@

network-visualizer-debug.o:
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} -c src/network_visualizer.cpp -o $@

neat_coro-valgrind: ai-debug.o neat_coro-debug.o network-visualizer-debug.o \
                    vendors/ethreads/threading-debug.o
	${CXX_DEBUG} ${CXXFLAGS_DEBUG} ${ZLIB} ${RAYLIB} ${ETHREADS_LIBS} $^ -o $@

# Helgrind targets
valgrind-all: neat-valgrind neat_coro-valgrind

helgrind-neat: neat-valgrind
	valgrind --tool=helgrind --suppressions=vendors/ethreads/valgrind.supp ./neat-valgrind

helgrind-coro: neat_coro-valgrind
	valgrind --tool=helgrind --suppressions=vendors/ethreads/valgrind.supp ./neat_coro-valgrind

#########################################################################################
# Profiling builds (optimized with frame pointers for accurate stack traces)
#########################################################################################

CFLAGS_PROFILE = -march=native -O3 -g -fno-omit-frame-pointer -Wall -Wextra -pedantic $(INC) $(ZLIB_CFLAGS)
CXXFLAGS_PROFILE = -std=c++20 $(CFLAGS_PROFILE)

brain-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c src/brain.cpp -o $@

mutation-rate-container-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c src/mutation_rate_container.cpp -o $@

pool-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c src/pool.cpp -o $@

speciating-parameter-container-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c src/speciating_parameter_container.cpp -o $@

model-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c src/model.cpp -o $@

ai-profile.o: brain-profile.o mutation-rate-container-profile.o pool-profile.o \
              speciating-parameter-container-profile.o model-profile.o
	ld -r $^ -o $@

vendors/ethreads/threading-profile.o:
	make -C vendors/ethreads threading-profile.o

neat-profile.o:
	${CXX} ${CXXFLAGS_PROFILE} -c builds/neat.cpp -o $@

neat-profile: ai-profile.o neat-profile.o vendors/ethreads/threading-profile.o
	${CXX} ${CXXFLAGS_PROFILE} ${ZLIB} ${ETHREADS_LIBS} $^ -o $@

# Run profiling and generate flamegraph + text report
# Uses frame pointer call-graph (faster than dwarf) with 10 second timeout
profile: neat-profile
	rm -f xor_pool xor_best perf.data
	timeout 10s perf record -F 997 -g --call-graph fp ./neat-profile || true
	perf script | stackcollapse-perf.pl | flamegraph.pl > profile.svg
	perf report --stdio --no-children > profile.txt
	@echo ""
	@echo "Generated:"
	@echo "  profile.svg - Flamegraph (open in browser)"
	@echo "  profile.txt - Text summary of hotspots"

clean:
	-rm -f neat neat_coro neat_genome_control neat_hebbian neat_coevolution neat_visualized
	-rm -f neat-valgrind neat_coro-valgrind neat-profile *.o *-debug.o *-profile.o
	-rm -f profile.svg profile.txt perf.data perf.data.old
	-rm -f mux_best_brain.txt
	make -C vendors/ethreads clean
	make -C vendors/exstd clean
	make -C vendors/emath clean
