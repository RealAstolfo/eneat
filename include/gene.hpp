#ifndef ENEAT_GENE_HPP
#define ENEAT_GENE_HPP

#include "functions.hpp"
#include "math.hpp"

struct gene {
  size_t innovation_num = -1;
  size_t from_node = -1;
  size_t to_node = -1;
  exfloat weight = 0.0f;
  ai_func_type activation = ai_func_type::RELU;
  bool enabled = true;
  bool is_bias_source = false;   // true if from_node is an evolved bias neuron
  bool is_recurrent = false;     // true if this is a recurrent connection
  bool is_time_delayed = false;  // true if this is a time-delayed recurrent (uses t-2)
  bool frozen = false;           // if true, weight cannot be mutated

  // Tracks cumulative mutations for compatibility distance calculation
  // (rtNEAT uses mutation_num instead of weight diff for speciation)
  exfloat mutation_num = 0.0f;

  // Reference to a trait for Hebbian learning parameters
  // 0 = no trait, 1+ = trait index in genome's trait vector
  size_t trait_id = 0;

  // Trait-derived parameters for this link (copied from trait during network build)
  static constexpr size_t NUM_TRAIT_PARAMS = 8;
  exfloat params[NUM_TRAIT_PARAMS] = {0.0f};
};

#endif
