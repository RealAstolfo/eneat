#ifndef ENEAT_GENOME_HPP
#define ENEAT_GENOME_HPP

#include <map>
#include <stdbool.h>
#include <stdlib.h>
#include <vector>

#include "gene.hpp"
#include "macros.h"
#include "mutation_rate_container.hpp"
#include "network_info_container.hpp"
#include "speciating_parameter_container.hpp"
#include "shared_state.hpp"
#include "trait.hpp"

typedef struct brain brain;

struct genome {
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> adjusted_fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> global_rank{0};

  // rtNEAT: Tracks how many generations this genome has existed
  // Used for maturity-based selection (only mature organisms can be replaced)
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> time_alive{0};

  size_t max_neuron;
  bool can_be_recurrent = false;
  mutation_rate_container mutation_rates;
  network_info_container network_info;
  std::map<size_t, gene> genes;

  // Traits for Hebbian learning parameters (shared across genes/neurons)
  std::vector<eneat::trait> traits;

  genome() = default;

  genome(const network_info_container &info,
         const mutation_rate_container &rates)
      : fitness(0), adjusted_fitness(0), global_rank(0) {
    mutation_rates = rates;
    network_info = info;
    max_neuron = info.functional_neurons;
  }

  // Copy constructor - sync_shared_value is not copyable, so we need explicit copy
  genome(const genome &other)
      : fitness(other.fitness.load()), adjusted_fitness(other.adjusted_fitness.load()),
        global_rank(other.global_rank.load()), time_alive(other.time_alive.load()),
        max_neuron(other.max_neuron), can_be_recurrent(other.can_be_recurrent),
        mutation_rates(other.mutation_rates), network_info(other.network_info),
        genes(other.genes), traits(other.traits) {}

  // Copy assignment
  genome &operator=(const genome &other) {
    if (this != &other) {
      fitness.store(other.fitness.load());
      adjusted_fitness.store(other.adjusted_fitness.load());
      global_rank.store(other.global_rank.load());
      time_alive.store(other.time_alive.load());
      max_neuron = other.max_neuron;
      can_be_recurrent = other.can_be_recurrent;
      mutation_rates = other.mutation_rates;
      network_info = other.network_info;
      genes = other.genes;
      traits = other.traits;
    }
    return *this;
  }

  // Move constructor
  genome(genome &&other) noexcept
      : fitness(other.fitness.load()), adjusted_fitness(other.adjusted_fitness.load()),
        global_rank(other.global_rank.load()), time_alive(other.time_alive.load()),
        max_neuron(other.max_neuron), can_be_recurrent(other.can_be_recurrent),
        mutation_rates(std::move(other.mutation_rates)),
        network_info(std::move(other.network_info)), genes(std::move(other.genes)),
        traits(std::move(other.traits)) {}

  // Move assignment
  genome &operator=(genome &&other) noexcept {
    if (this != &other) {
      fitness.store(other.fitness.load());
      adjusted_fitness.store(other.adjusted_fitness.load());
      global_rank.store(other.global_rank.load());
      time_alive.store(other.time_alive.load());
      max_neuron = other.max_neuron;
      can_be_recurrent = other.can_be_recurrent;
      mutation_rates = std::move(other.mutation_rates);
      network_info = std::move(other.network_info);
      genes = std::move(other.genes);
      traits = std::move(other.traits);
    }
    return *this;
  }

  // Get a trait by ID (returns nullptr if not found)
  const eneat::trait* get_trait(size_t trait_id) const {
    if (trait_id == 0 || trait_id > traits.size()) return nullptr;
    return &traits[trait_id - 1]; // trait_id is 1-indexed
  }

  eneat::trait* get_trait(size_t trait_id) {
    if (trait_id == 0 || trait_id > traits.size()) return nullptr;
    return &traits[trait_id - 1];
  }

  // Add a new trait and return its ID
  size_t add_trait(const eneat::trait& t) {
    eneat::trait new_trait = t;
    new_trait.trait_id = traits.size() + 1;
    traits.push_back(new_trait);
    return new_trait.trait_id;
  }

  // Increment time_alive (called each generation)
  void age() {
    time_alive.store(time_alive.load() + 1);
  }
};

#endif
