#include "mutation_rate_container.hpp"
#include "math.hpp"

#include <iostream>

std::istream &operator>>(std::istream &input, mutation_rate_container &mrc) {
  input >> mrc.connection_mutate_chance;
  input >> mrc.perturb_chance;
  input >> mrc.crossover_chance;
  input >> mrc.link_mutation_chance;
  input >> mrc.neuron_mutation_chance;
  input >> mrc.bias_mutation_chance;
  input >> mrc.step_size;
  input >> mrc.disable_mutation_chance;
  input >> mrc.enable_mutation_chance;
  input >> mrc.activation_mutation_chance;
  return input;
}

std::ostream &operator<<(std::ostream &output, mutation_rate_container &mrc) {
  output << mrc.connection_mutate_chance << std::endl;
  output << mrc.perturb_chance << std::endl;
  output << mrc.crossover_chance << std::endl;
  output << mrc.link_mutation_chance << std::endl;
  output << mrc.neuron_mutation_chance << std::endl;
  output << mrc.bias_mutation_chance << std::endl;
  output << mrc.step_size << std::endl;
  output << mrc.disable_mutation_chance << std::endl;
  output << mrc.enable_mutation_chance << std::endl;
  output << mrc.activation_mutation_chance << std::endl;
  return output;
}
