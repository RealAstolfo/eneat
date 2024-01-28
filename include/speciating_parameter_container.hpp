#ifndef ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP
#define ENEAT_SPECIATING_PARAMETER_CONTAINER_HPP

#include <iostream>
#include <stdlib.h>

struct speciating_parameter_container {
  size_t population = 240;
  size_t stale_species = 15;
  float delta_disjoint = 2.0f;
  float delta_weights = 0.4f;
  float delta_threshold = 1.3f;
};

std::istream &operator>>(std::istream &input,
                         speciating_parameter_container &spc);
std::ostream &operator<<(std::ostream &output,
                         speciating_parameter_container &spc);

#endif
