#ifndef ENEAT_GENOME_HPP
#define ENEAT_GENOME_HPP

#include <map>
#include <random>
#include <set>
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

  // rtNEAT: Node traits - maps node_id to trait_id (0 = no trait)
  std::map<size_t, size_t> node_traits;

  // rtNEAT: Frozen nodes - nodes that cannot have their traits mutated
  std::set<size_t> frozen_nodes;

  // rtNEAT: Origin tracking (for debugging and adaptive strategies)
  bool mate_baby = false;        // Created via crossover
  bool mut_struct_baby = false;  // Created via structural mutation

  // rtNEAT: Super champion tracking
  size_t super_champ_offspring = 0;  // Reserved offspring for population champion
  bool pop_champ = false;            // Is this the population champion?

  // rtNEAT: Organism metadata (from reference organism.h)
  size_t orig_fitness = 0;       // Original fitness before adjustment
  size_t generation_born = 0;    // Generation this organism was born
  bool winner = false;           // Win marker for competitive scenarios
  bool eliminate = false;        // Marked for destruction
  bool champion = false;         // Species champion flag

  // rtNEAT: Extended organism metadata
  exfloat error = 0.0f;          // Error value for competitive coevolution
  size_t high_fit = 0;           // Highest fitness ever achieved by this organism
  bool pop_champ_child = false;  // Is this a child of the population champion?

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
        genes(other.genes), traits(other.traits),
        node_traits(other.node_traits), frozen_nodes(other.frozen_nodes),
        mate_baby(other.mate_baby), mut_struct_baby(other.mut_struct_baby),
        super_champ_offspring(other.super_champ_offspring), pop_champ(other.pop_champ),
        orig_fitness(other.orig_fitness), generation_born(other.generation_born),
        winner(other.winner), eliminate(other.eliminate), champion(other.champion) {}

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
      node_traits = other.node_traits;
      frozen_nodes = other.frozen_nodes;
      mate_baby = other.mate_baby;
      mut_struct_baby = other.mut_struct_baby;
      super_champ_offspring = other.super_champ_offspring;
      pop_champ = other.pop_champ;
      orig_fitness = other.orig_fitness;
      generation_born = other.generation_born;
      winner = other.winner;
      eliminate = other.eliminate;
      champion = other.champion;
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
        traits(std::move(other.traits)),
        node_traits(std::move(other.node_traits)), frozen_nodes(std::move(other.frozen_nodes)),
        mate_baby(other.mate_baby), mut_struct_baby(other.mut_struct_baby),
        super_champ_offspring(other.super_champ_offspring), pop_champ(other.pop_champ),
        orig_fitness(other.orig_fitness), generation_born(other.generation_born),
        winner(other.winner), eliminate(other.eliminate), champion(other.champion) {}

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
      node_traits = std::move(other.node_traits);
      frozen_nodes = std::move(other.frozen_nodes);
      mate_baby = other.mate_baby;
      mut_struct_baby = other.mut_struct_baby;
      super_champ_offspring = other.super_champ_offspring;
      pop_champ = other.pop_champ;
      orig_fitness = other.orig_fitness;
      generation_born = other.generation_born;
      winner = other.winner;
      eliminate = other.eliminate;
      champion = other.champion;
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

  // rtNEAT utility: Count non-disabled genes (extrons)
  // Based on rtNEAT genome.h:169
  size_t extrons() const {
    size_t count = 0;
    for (const auto& [_, g] : genes) {
      if (g.enabled) count++;
    }
    return count;
  }

  // rtNEAT utility: Randomize trait assignments for links and nodes
  // Based on rtNEAT genome.h:172
  void randomize_traits() {
    if (traits.empty()) return;

    // Use thread_local generator for randomization
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<size_t> trait_dist(1, traits.size());

    // Randomize link traits
    for (auto& [_, g] : genes) {
      g.trait_id = trait_dist(gen);
    }

    // Randomize node traits
    for (auto& [node_id, _] : node_traits) {
      node_traits[node_id] = trait_dist(gen);
    }
  }
};

#endif
