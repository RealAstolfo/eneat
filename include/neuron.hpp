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
  bool is_time_delayed = false;  // Time-delayed recurrent (uses last_activation2)

  // Trait-derived parameters for this link (copied from trait on network build)
  static constexpr size_t NUM_TRAIT_PARAMS = 8;
  exfloat params[NUM_TRAIT_PARAMS] = {0.0f};

  // For future learning: accumulated weight changes
  exfloat added_weight = 0.0f;

  neuron_connection() : from_neuron(0), weight(0.0f), trait_id(0), is_recurrent(false), is_time_delayed(false), added_weight(0.0f) {}
  neuron_connection(size_t from, exfloat w, size_t tid = 0, bool rec = false, bool td = false)
      : from_neuron(from), weight(w), trait_id(tid), is_recurrent(rec), is_time_delayed(td), added_weight(0.0f) {}
};

struct neuron {
  neuron_place type = HIDDEN;
  exfloat value = 0.0f;
  exfloat last_activation = 0.0f;   // For Hebbian learning: previous activation (t-1)
  exfloat last_activation2 = 0.0f;  // For time-delayed recurrent: activation at (t-2)
  exfloat activation_count = 0.0f;  // For averaging activations
  bool visited = false;
  std::vector<neuron_connection> in_connections;
  ai_func_type activation_function = RELU;
  size_t trait_id = 0;  // Node trait for node-level learning parameters

  // Trait-derived parameters for this node (copied from trait on network build)
  // params[0-7] used for various learning/activation parameters
  static constexpr size_t NUM_TRAIT_PARAMS = 8;
  exfloat params[NUM_TRAIT_PARAMS] = {0.0f};

  // rtNEAT: Override capabilities for forcing activation values
  // Reference: nnode.cpp:334-347
  exfloat override_value = 0.0f;  // Value to use when overridden
  bool override_active = false;   // Whether override is active

  neuron() {}
  ~neuron() { in_connections.clear(); }

  // Force an output value on this neuron
  void override_output(exfloat new_output) {
    override_value = new_output;
    override_active = true;
  }

  // Check if this neuron is being overridden
  bool overridden() const {
    return override_active;
  }

  // Apply the override value to the neuron's activation
  void activate_override() {
    if (override_active) {
      value = override_value;
    }
  }

  // Clear the override
  void clear_override() {
    override_active = false;
    override_value = 0.0f;
  }

  // rtNEAT: Sensor load with time-delay memory shifting
  // Reference: nnode.cpp:66-88
  bool sensor_load(exfloat input_value) {
    if (type == INPUT || type == BIAS) {
      // Shift activation history
      last_activation2 = last_activation;
      last_activation = value;
      activation_count += 1.0f;
      value = input_value;
      return true;
    }
    return false;
  }

  // Get active output (returns value if activated, else 0)
  exfloat get_active_out() const {
    if (activation_count > 0)
      return value;
    return 0.0f;
  }

  // Get time-delayed active output (previous timestep)
  exfloat get_active_out_td() const {
    if (activation_count > 1)
      return last_activation;
    return 0.0f;
  }
};

#endif
