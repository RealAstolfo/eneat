#ifndef ENEAT_MUTATION_RATE_CONTAINER_HPP
#define ENEAT_MUTATION_RATE_CONTAINER_HPP

#include <iostream>

#include "math.hpp"

struct mutation_rate_container {
  // Weight mutation (aligned with rtNEAT reference params256.ne)
  exfloat connection_mutate_chance = 0.9f;    // mutate_link_weights_prob (was 0.75f)
  exfloat perturb_chance = 0.9f;              // Perturb vs reset weight (within mutation)
  exfloat step_size = 2.5f;                   // weigh_mut_power (was 0.1f)

  // Structural mutations (aligned with rtNEAT reference)
  exfloat link_mutation_chance = 0.1f;        // mutate_add_link_prob (was 2.0f)
  exfloat neuron_mutation_chance = 0.0025f;   // mutate_add_node_prob (was 0.05f)
  exfloat bias_mutation_chance = 0.4f;        // Chance to add connection FROM existing bias
  exfloat bias_neuron_mutation_chance = 0.02f;// Chance to add NEW bias neuron

  // Gene enable/disable (disabled by default like reference)
  exfloat disable_mutation_chance = 0.0f;     // mutate_toggle_enable_prob (was 0.4f)
  exfloat enable_mutation_chance = 0.0f;      // mutate_gene_reenable_prob (was 0.2f)

  // Activation function mutation
  exfloat activation_mutation_chance = 0.07f;

  // Crossover types (aligned with rtNEAT reference)
  exfloat crossover_chance = 0.75f;           // 1 - mutate_only_prob
  exfloat mutate_only_prob = 0.25f;           // mutate_only_prob
  exfloat multipoint_avg_chance = 0.4f;       // mate_multipoint_avg_prob (was 0.25f)
  exfloat singlepoint_chance = 0.0f;          // mate_singlepoint_prob (was 0.10f)

  // Trait mutations (aligned with rtNEAT reference)
  exfloat trait_mutation_chance = 0.1f;       // mutate_random_trait_prob
  exfloat trait_param_mutation_power = 1.0f;  // trait_mutation_power (was 0.1f)
  exfloat link_trait_mutation_chance = 0.1f;  // mutate_link_trait_prob
  exfloat node_trait_mutation_chance = 0.1f;  // mutate_node_trait_prob

  // Recurrence
  exfloat recur_only_prob = 0.2f;             // recur_prob
};

std::istream &operator>>(std::istream &input, mutation_rate_container &mrc);
std::ostream &operator<<(std::ostream &output, mutation_rate_container &mrc);

#endif
