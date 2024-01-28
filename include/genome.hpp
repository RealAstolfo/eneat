#ifndef ENEAT_GENOME_HPP
#define ENEAT_GENOME_HPP

#include <map>
#include <stdbool.h>
#include <stdlib.h>

#include "gene.hpp"
#include "macros.h"
#include "mutation_rate_container.hpp"
#include "network_info_container.hpp"
#include "speciating_parameter_container.hpp"

typedef struct brain brain;

struct genome {
  size_t fitness = 0;
  size_t adjusted_fitness = 0;
  size_t global_rank = 0;
  size_t max_neuron;
  bool can_be_recurrent = false;
  mutation_rate_container mutation_rates;
  network_info_container network_info;
  std::map<size_t, gene> genes;

  genome() {}

  genome(const network_info_container &info,
         const mutation_rate_container &rates) {
    mutation_rates = rates;
    network_info = info;
    max_neuron = info.functional_neurons;
  }
};

#endif
