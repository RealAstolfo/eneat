#ifndef ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP
#define ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP

#include <iostream>
#include <stdlib.h>

struct speciating_parameter_container {
  size_t population = 240;
  size_t stale_species = 15;        // Generations without improvement before extinction

  // Compatibility distance coefficients (rtNEAT-style)
  float delta_disjoint = 2.0f;      // Weight for disjoint genes (within range)
  float delta_excess = 2.0f;        // Weight for excess genes (beyond range)
  float delta_weights = 0.4f;       // Weight for mutation_num difference
  float delta_threshold = 1.3f;     // Species compatibility threshold

  // rtNEAT parameters
  size_t time_alive_minimum = 5;    // Minimum generations before selection eligibility
  float survival_thresh = 0.2f;     // Fraction of species that survives each gen (top 20%)
  size_t dropoff_age = 15;          // Generations without improvement before heavy penalty
  float age_significance = 1.0f;    // Fitness multiplier for young species (< 10 gens)

  // Interspecies mating
  float interspecies_mate_rate = 0.001f;  // Probability of mating outside species
};

std::istream &operator>>(std::istream &input,
                         speciating_parameter_container &spc);
std::ostream &operator<<(std::ostream &output,
                         speciating_parameter_container &spc);

#endif
