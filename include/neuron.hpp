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

// Connection structure for Hebbian learning support
struct neuron_connection {
  size_t from_neuron;         // Source neuron index
  exfloat weight;             // Connection weight
  size_t trait_id = 0;        // Trait for Hebbian learning (0 = no trait)
  bool is_recurrent = false;  // Recurrent connection flag

  neuron_connection() : from_neuron(0), weight(0.0f), trait_id(0), is_recurrent(false) {}
  neuron_connection(size_t from, exfloat w, size_t tid = 0, bool rec = false)
      : from_neuron(from), weight(w), trait_id(tid), is_recurrent(rec) {}
};

struct neuron {
  neuron_place type = HIDDEN;
  exfloat value = 0.0f;
  exfloat last_activation = 0.0f;  // For Hebbian learning: previous activation
  exfloat activation_count = 0.0f; // For averaging activations
  bool visited = false;
  std::vector<neuron_connection> in_connections;
  ai_func_type activation_function = RELU;
  size_t trait_id = 0;  // Node trait for node-level learning parameters

  neuron() {}
  ~neuron() { in_connections.clear(); }
};

#endif
