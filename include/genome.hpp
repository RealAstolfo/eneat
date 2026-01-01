#ifndef ENEAT_GENOME_HPP
#define ENEAT_GENOME_HPP

#include <atomic>
#include <map>
#include <stdbool.h>
#include <stdlib.h>
#include <vector>

#include "gene.hpp"
#include "macros.h"
#include "mutation_rate_container.hpp"
#include "network_info_container.hpp"
#include "speciating_parameter_container.hpp"

typedef struct brain brain;

struct genome {
  std::atomic<size_t> fitness{0};
  std::atomic<size_t> adjusted_fitness{0};
  std::atomic<size_t> global_rank{0};
  size_t max_neuron;
  bool can_be_recurrent = false;
  mutation_rate_container mutation_rates;
  network_info_container network_info;
  std::map<size_t, gene> genes;

  genome() : fitness(0), adjusted_fitness(0), global_rank(0) {}

  genome(const network_info_container &info,
         const mutation_rate_container &rates)
      : fitness(0), adjusted_fitness(0), global_rank(0) {
    mutation_rates = rates;
    network_info = info;
    max_neuron = info.functional_neurons;
  }

  // Copy constructor - atomics are not copyable, so we need explicit copy
  genome(const genome &other)
      : fitness(other.fitness.load()), adjusted_fitness(other.adjusted_fitness.load()),
        global_rank(other.global_rank.load()), max_neuron(other.max_neuron),
        can_be_recurrent(other.can_be_recurrent), mutation_rates(other.mutation_rates),
        network_info(other.network_info), genes(other.genes) {}

  // Copy assignment
  genome &operator=(const genome &other) {
    if (this != &other) {
      fitness.store(other.fitness.load());
      adjusted_fitness.store(other.adjusted_fitness.load());
      global_rank.store(other.global_rank.load());
      max_neuron = other.max_neuron;
      can_be_recurrent = other.can_be_recurrent;
      mutation_rates = other.mutation_rates;
      network_info = other.network_info;
      genes = other.genes;
    }
    return *this;
  }

  // Move constructor
  genome(genome &&other) noexcept
      : fitness(other.fitness.load()), adjusted_fitness(other.adjusted_fitness.load()),
        global_rank(other.global_rank.load()), max_neuron(other.max_neuron),
        can_be_recurrent(other.can_be_recurrent),
        mutation_rates(std::move(other.mutation_rates)),
        network_info(std::move(other.network_info)), genes(std::move(other.genes)) {}

  // Move assignment
  genome &operator=(genome &&other) noexcept {
    if (this != &other) {
      fitness.store(other.fitness.load());
      adjusted_fitness.store(other.adjusted_fitness.load());
      global_rank.store(other.global_rank.load());
      max_neuron = other.max_neuron;
      can_be_recurrent = other.can_be_recurrent;
      mutation_rates = std::move(other.mutation_rates);
      network_info = std::move(other.network_info);
      genes = std::move(other.genes);
    }
    return *this;
  }
};

#endif
