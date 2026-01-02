#ifndef ENEAT_POOL_HPP
#define ENEAT_POOL_HPP

#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <stdbool.h>
#include <utility>
#include <vector>

#include "coro_task.hpp"
#include "innovation_channel.hpp"
#include "mutation_rate_container.hpp"
#include "network_info_container.hpp"
#include "shared_state.hpp"
#include "speciating_parameter_container.hpp"
#include "species_channel.hpp"

typedef struct specie specie;
typedef struct genome genome;

// Thread-local RNG for lock-free random number generation
namespace eneat {
inline thread_local std::mt19937 tl_generator{std::random_device{}()};

template <typename Dist>
inline auto get_rand(Dist &d) -> decltype(d(tl_generator)) {
  return d(tl_generator);
}
} // namespace eneat

// Crossover type enum
enum class crossover_type {
  MULTIPOINT,       // Random selection from matching genes (default)
  MULTIPOINT_AVG,   // Average weights of matching genes
  SINGLEPOINT       // Traditional single-point crossover
};

struct pool {
  // Channel-based innovation tracking
  eneat::innovation_channel innovation_chan;

  // Channel-based species management
  std::unique_ptr<eneat::species_channel> species_chan;

  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> max_fitness{0};
  mutation_rate_container mutation_rates;
  speciating_parameter_container speciating_parameters;
  network_info_container network_info;
  std::list<specie> species;

  // Counter for unique species IDs
  size_t next_species_id = 1;

  pool(size_t input, size_t output, size_t population = 150,
       size_t bias = 1, bool rec = false);

  // Compatibility distance calculation
  bool is_same_species(const genome &g1, const genome &g2);
  exfloat disjoint(const genome &g1, const genome &g2);
  exfloat excess(const genome &g1, const genome &g2);
  exfloat mut_diff(const genome &g1, const genome &g2);

  // Get all genomes as (species, genome) pairs (UNSAFE - pointers may be invalidated)
  std::vector<std::pair<specie *, genome *>> get_genomes();

  // Safe index-based genome access
  // Returns (species_index, genome_index) pairs - indices remain valid across co_await
  std::vector<std::pair<size_t, size_t>> get_genome_indices();

  // Get total number of genomes
  size_t get_population_size();

  // Safely access a genome by indices with proper locking
  // Returns a copy of the genome for safe use across co_await points
  genome get_genome_copy(size_t species_idx, size_t genome_idx);

  // Update a genome's fitness by indices (thread-safe)
  void set_genome_fitness(size_t species_idx, size_t genome_idx, size_t fitness);

  // Increment a genome's time_alive by indices (thread-safe)
  void increment_genome_age(size_t species_idx, size_t genome_idx);

  // Crossover types
  genome crossover(const genome &g1, const genome &g2);
  genome crossover_multipoint(const genome &g1, const genome &g2);
  genome crossover_multipoint_avg(const genome &g1, const genome &g2);
  genome crossover_singlepoint(const genome &g1, const genome &g2);

  // Mutation functions
  ethreads::coro_task<void> mutate_activation(genome &g);
  ethreads::coro_task<void> mutate_weight(genome &g);
  ethreads::coro_task<void> mutate_enable_disable(genome &g, const bool &enable);
  ethreads::coro_task<void> mutate_link(genome &g, const bool &force_bias);
  ethreads::coro_task<void> mutate_neuron(genome &g);
  ethreads::coro_task<void> mutate_bias_neuron(genome &g);
  ethreads::coro_task<void> mutate(genome &g);

  // Trait mutations (Hebbian learning)
  ethreads::coro_task<void> mutate_random_trait(genome &g);
  ethreads::coro_task<void> mutate_link_trait(genome &g);
  ethreads::coro_task<void> mutate_node_trait(genome &g);

  // Synchronous mutation for initialization
  void mutate_sync(genome &g);

  // Weight-only mutation (for super champion offspring - 80% weight only, 20% structural)
  void mutate_weight_only(genome &g);

  // rtNEAT: Add sensor mutation - connects unconnected input sensors to outputs
  // Reference: genome.cpp:1804 (mutate_add_sensor)
  ethreads::coro_task<void> mutate_add_sensor(genome &g);
  void mutate_add_sensor_sync(genome &g);

  // Species management
  void add_to_species(const genome &child);
  ethreads::coro_task<void> add_to_species_async(genome child);
  void remove_stale_species();
  void remove_weak_species();

  // Fitness calculation
  void calculate_average_fitness(specie &s);
  void adjust_species_fitness(specie &s);
  size_t total_average_fitness();

  // Breeding
  ethreads::coro_task<genome> breed_child(specie &s);

  // Real-time evolution
  specie* choose_parent_species();
  ethreads::coro_task<genome> reproduce_one_async();
  ethreads::coro_task<bool> remove_worst_async();
  void estimate_all_averages();
  void age_all_organisms();

  // Synchronous versions (avoid coroutine scheduling overhead)
  genome reproduce_one();
  bool remove_worst();
  bool remove_worst_probabilistic();  // Fitness-proportional probabilistic removal
  genome breed_child_sync(specie &s);

  // Initialize species channel (call before async operations)
  void init_species_channel();

  // rtNEAT reference features

  // Delta-coding: emergency diversity injection when population stagnates
  size_t highest_fitness_ever = 0;
  size_t generations_since_improvement = 0;
  void check_delta_coding();
  void apply_delta_coding();

  // Babies stolen: offspring redistribution from weak to strong species
  void redistribute_offspring();

  // Dynamic compatibility threshold: auto-adjust to maintain target species count
  size_t offspring_since_compat_adjust = 0;
  void adjust_compatibility_threshold();
  void reassign_all_species();

  // Expected offspring calculation (proportional to species fitness)
  void calculate_expected_offspring();
  void reset_champion_flags();  // Reset champion_preserved flags for new generation

  // Interspecies mating helpers
  specie* choose_random_species_excluding(const specie& exclude);
  genome get_species_champion(const specie& s);

  // Network complexity analysis
  size_t calculate_network_depth(const genome& g);

  // Genome verification (debug/validation)
  bool verify_genome(const genome& g) const;
};

std::istream &operator>>(std::istream &input, pool &p);
std::ostream &operator<<(std::ostream &output, pool &p);

#endif
