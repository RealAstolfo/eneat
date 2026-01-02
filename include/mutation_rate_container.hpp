#ifndef ENEAT_MUTATION_RATE_CONTAINER_HPP
#define ENEAT_MUTATION_RATE_CONTAINER_HPP

#include <iostream>

#include "math.hpp"

struct mutation_rate_container {
  // Weight mutation
  exfloat connection_mutate_chance = 0.75f;   // Probability of mutating weights
  exfloat perturb_chance = 0.9f;              // Perturb vs reset weight (within mutation)
  exfloat step_size = 0.1f;                   // Weight mutation magnitude

  // Structural mutations
  exfloat link_mutation_chance = 2.0f;        // Expected # of new links per generation
  exfloat neuron_mutation_chance = 0.05f;     // Probability of adding a node
  exfloat bias_mutation_chance = 0.4f;        // Chance to add connection FROM existing bias
  exfloat bias_neuron_mutation_chance = 0.02f;// Chance to add NEW bias neuron

  // Gene enable/disable
  exfloat disable_mutation_chance = 0.4f;
  exfloat enable_mutation_chance = 0.2f;

  // Activation function mutation
  exfloat activation_mutation_chance = 0.07f;

  // Crossover types (rtNEAT-style multiple crossover)
  exfloat crossover_chance = 0.75f;           // Probability of crossover vs mutation-only
  exfloat mutate_only_prob = 0.25f;           // Probability of mutation without crossover
  exfloat multipoint_avg_chance = 0.25f;      // Use averaging for matching genes
  exfloat singlepoint_chance = 0.10f;         // Use single-point crossover

  // Trait mutations (for Hebbian learning)
  exfloat trait_mutation_chance = 0.1f;       // Probability of mutating traits
  exfloat trait_param_mutation_power = 0.1f;  // Trait parameter mutation magnitude
  exfloat link_trait_mutation_chance = 0.1f;  // Change link's trait assignment
  exfloat node_trait_mutation_chance = 0.1f;  // Change node's trait assignment

  // Recurrence
  exfloat recur_only_prob = 0.2f;             // Probability of adding recurrent link
};

std::istream &operator>>(std::istream &input, mutation_rate_container &mrc);
std::ostream &operator<<(std::ostream &output, mutation_rate_container &mrc);

#endif
