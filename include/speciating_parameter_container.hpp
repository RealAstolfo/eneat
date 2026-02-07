#ifndef ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP
#define ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP

#include <iostream>
#include <stdlib.h>

struct speciating_parameter_container {
  size_t population = 256;          // pop_size (was 240)
  size_t stale_species = 15;        // Generations without improvement before extinction

  // Compatibility distance coefficients (aligned with rtNEAT reference)
  float delta_disjoint = 1.0f;      // disjoint_coeff (was 2.0f)
  float delta_excess = 1.0f;        // excess_coeff (was 2.0f)
  float delta_weights = 2.0f;       // mutdiff_coeff (was 0.4f)
  float delta_threshold = 3.0f;     // compat_thresh (was 1.3f)

  // rtNEAT parameters (aligned with reference)
  size_t time_alive_minimum = 5;    // Minimum generations before selection eligibility
  float survival_thresh = 0.2f;     // Fraction of species that survives each gen (top 20%)
  size_t dropoff_age = 1000;        // dropoff_age (was 15)
  float age_significance = 1.0f;    // Fitness multiplier for young species (< 10 gens)

  // Interspecies mating
  float interspecies_mate_rate = 0.05f;   // interspecies_mate_rate (was 0.001f)

  // Delta-coding parameters (for population stagnation recovery)
  bool delta_coding_enabled = true;

  // Babies stolen parameters (offspring redistribution to strong species)
  size_t babies_stolen = 5;         // Number of offspring to redistribute (0 = disabled)

  // Dynamic compatibility threshold parameters
  bool dynamic_threshold_enabled = true;
  size_t target_species_count = 10;       // Desired number of species
  size_t compat_adjust_frequency = 10;    // How often to adjust (every N offspring)
  float compat_threshold_delta = 0.1f;    // Adjustment step size
  float compat_threshold_min = 0.3f;      // Minimum threshold

  // Link mutation retry parameter
  size_t newlink_tries = 20;              // Attempts to find valid connection
};

std::istream &operator>>(std::istream &input,
                         speciating_parameter_container &spc);
std::ostream &operator<<(std::ostream &output,
                         speciating_parameter_container &spc);

#endif
