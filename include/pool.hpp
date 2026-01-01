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

struct pool {
  // Channel-based innovation tracking (replaces mutex-protected container)
  eneat::innovation_channel innovation_chan;

  // Channel-based species management (lazy initialized)
  std::unique_ptr<eneat::species_channel> species_chan;

  ethreads::sync_shared_value<size_t> generation_number{1};
  ethreads::sync_shared_value<size_t> max_fitness{0};
  mutation_rate_container mutation_rates;
  speciating_parameter_container speciating_parameters;
  network_info_container network_info;
  std::list<specie> species;

  pool(size_t input, size_t output, size_t bias = 1, bool rec = false);

  bool is_same_species(const genome &g1, const genome &g2);
  void new_generation();
  std::vector<std::pair<specie *, genome *>> get_genomes();
  genome crossover(const genome &g1, const genome &g2);
  // Coroutine mutation functions
  ethreads::coro_task<void> mutate_activation(genome &g);
  ethreads::coro_task<void> mutate_weight(genome &g);
  ethreads::coro_task<void> mutate_enable_disable(genome &g, const bool &enable);
  ethreads::coro_task<void> mutate_link(genome &g, const bool &force_bias);
  ethreads::coro_task<void> mutate_neuron(genome &g);
  ethreads::coro_task<void> mutate_bias_neuron(genome &g);  // Add new bias neuron
  ethreads::coro_task<void> mutate(genome &g);

  // Synchronous mutation for single-threaded initialization
  void mutate_sync(genome &g);
  exfloat disjoint(const genome &g1, const genome &g2);
  exfloat weights(const genome &g1, const genome &g2);
  void rank_globally();
  void calculate_average_fitness(specie &s);
  size_t total_average_fitness();
  void cull_species(const bool &cut_to_one);
  ethreads::coro_task<genome> breed_child(specie &s);
  void remove_stale_species();
  void remove_weak_species();
  void add_to_species(const genome &child);

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

  // Deprecated: use eneat::get_rand() instead
  // Kept for compatibility during transition
  template <typename Dist>
  inline auto get_rand(Dist &d, std::mt19937 &) -> decltype(d(eneat::tl_generator)) {
    return eneat::get_rand(d);
  }
};

std::istream &operator>>(std::istream &input, pool &p);
std::ostream &operator<<(std::ostream &output, pool &p);

#endif
