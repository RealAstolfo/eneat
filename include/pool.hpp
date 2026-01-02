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
// Each thread gets its own generator seeded from random_device
namespace eneat {
inline thread_local std::mt19937 tl_generator{std::random_device{}()};

template <typename Dist>
inline auto get_rand(Dist &d) -> decltype(d(tl_generator)) {
  return d(tl_generator);
}
} // namespace eneat

// Crossover type enum for multiple crossover strategies
enum class crossover_type {
  MULTIPOINT,       // Random selection from matching genes (default)
  MULTIPOINT_AVG,   // Average weights of matching genes
  SINGLEPOINT       // Traditional single-point crossover
};

struct pool {
  // Channel-based innovation tracking (replaces mutex-protected container)
  eneat::innovation_channel innovation_chan;

  // Channel-based species management (lazy initialized)
  std::unique_ptr<eneat::species_channel> species_chan;

  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> generation_number{1};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> max_fitness{0};
  mutation_rate_container mutation_rates;
  speciating_parameter_container speciating_parameters;
  network_info_container network_info;
  std::list<specie> species;

  // rtNEAT: Counter for unique species IDs
  size_t next_species_id = 1;

  pool(size_t input, size_t output, size_t bias = 1, bool rec = false);

  // Compatibility distance calculation (rtNEAT-style with disjoint/excess separation)
  bool is_same_species(const genome &g1, const genome &g2);
  exfloat disjoint(const genome &g1, const genome &g2);  // Genes within range
  exfloat excess(const genome &g1, const genome &g2);    // Genes beyond range
  exfloat mut_diff(const genome &g1, const genome &g2);  // Mutation number difference

  // Generation management
  void new_generation();
  std::vector<std::pair<specie *, genome *>> get_genomes();

  // Multiple crossover types (rtNEAT-style)
  genome crossover(const genome &g1, const genome &g2);  // Default multipoint
  genome crossover_multipoint(const genome &g1, const genome &g2);
  genome crossover_multipoint_avg(const genome &g1, const genome &g2);
  genome crossover_singlepoint(const genome &g1, const genome &g2);

  // Coroutine mutation functions
  ethreads::coro_task<void> mutate_activation(genome &g);
  ethreads::coro_task<void> mutate_weight(genome &g);
  ethreads::coro_task<void> mutate_enable_disable(genome &g, const bool &enable);
  ethreads::coro_task<void> mutate_link(genome &g, const bool &force_bias);
  ethreads::coro_task<void> mutate_neuron(genome &g);
  ethreads::coro_task<void> mutate_bias_neuron(genome &g);  // Add new bias neuron
  ethreads::coro_task<void> mutate(genome &g);

  // Trait mutations (for Hebbian learning)
  ethreads::coro_task<void> mutate_random_trait(genome &g);
  ethreads::coro_task<void> mutate_link_trait(genome &g);

  // Synchronous mutation for single-threaded initialization
  void mutate_sync(genome &g);

  // Fitness and selection
  void rank_globally();
  void calculate_average_fitness(specie &s);
  size_t total_average_fitness();
  void cull_species(const bool &cut_to_one);
  ethreads::coro_task<genome> breed_child(specie &s);
  void remove_stale_species();
  void remove_weak_species();
  void add_to_species(const genome &child);

  // rtNEAT fitness adjustment (with age penalties/bonuses)
  void adjust_species_fitness(specie &s);

  // rtNEAT: Real-time evolution methods
  specie* choose_parent_species();                        // Roulette wheel species selection
  ethreads::coro_task<genome> reproduce_one_async();      // Produce single offspring
  ethreads::coro_task<void> remove_worst_async();         // Remove lowest fitness organism
  void estimate_all_averages();                           // Update running averages

  // Age all organisms (increment time_alive)
  void age_all_organisms();

  // Initialize species channel (call before async operations)
  void init_species_channel();
  // Async add to species via channel
  ethreads::coro_task<void> add_to_species_async(genome child);

  // Async versions of species operations for parallel execution
  ethreads::coro_task<void> cull_species_async(const bool &cut_to_one);
  ethreads::coro_task<void> rank_globally_async();
  ethreads::coro_task<void> calculate_all_average_fitness_async();
  // Async new_generation using when_all for parallel operations
  ethreads::coro_task<void> new_generation_async();
};

std::istream &operator>>(std::istream &input, pool &p);
std::ostream &operator<<(std::ostream &output, pool &p);

#endif
