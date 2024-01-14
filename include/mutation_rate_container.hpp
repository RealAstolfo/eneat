#ifndef MUTATION_RATE_CONTAINER_HPP
#define MUTATION_RATE_CONTAINER_HPP

#include "math.hpp"
#include <iostream>

struct mutation_rate_container {
  exfloat connection_mutate_chance = 0.75f;
  exfloat perturb_chance = 0.9f;
  exfloat crossover_chance = 0.75f;
  exfloat link_mutation_chance = 2.0f;
  exfloat neuron_mutation_chance = 0.05f;
  exfloat bias_mutation_chance = 0.4f;
  exfloat step_size = 0.1f;
  exfloat disable_mutation_chance = 0.4f;
  exfloat enable_mutation_chance = 0.2f;
  exfloat activation_mutation_chance = 0.25f;
};

std::istream &operator>>(std::istream &input, mutation_rate_container &mrc);
std::ostream &operator<<(std::ostream &output, mutation_rate_container &mrc);

#endif
