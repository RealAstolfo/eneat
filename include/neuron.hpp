#ifndef ENEAT_NEURON_HPP
#define ENEAT_NEURON_HPP

#include <stdbool.h>
#include <stdlib.h>
#include <utility>
#include <vector>

#include "functions.hpp"

enum neuron_type {
  RECURRENT = 0,
  NON_RECURRENT = 1,
};

enum neuron_place {
  HIDDEN = 0,
  INPUT = 1,
  OUTPUT = 2,
  BIAS = 3,
};

struct neuron {
  neuron_place type = HIDDEN;
  exfloat value = 0.0f;
  bool visited = false;
  std::vector<std::pair<size_t, exfloat>> in_neurons;
  ai_func_type activation_function = RELU;

  neuron() {}
  ~neuron() { in_neurons.clear(); }
};

#endif
