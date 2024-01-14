#include "speciating_parameter_container.hpp"
#include <iostream>
#include <sys/types.h>

std::istream &operator>>(std::istream &input,
                         speciating_parameter_container &spc) {
  input >> spc.population;
  input >> spc.delta_disjoint;
  input >> spc.delta_weights;
  input >> spc.delta_threshold;
  input >> spc.stale_species;
  return input;
}

std::ostream &operator<<(std::ostream &output,
                         speciating_parameter_container &spc) {
  output << spc.population << std::endl;
  output << spc.delta_disjoint << std::endl;
  output << spc.delta_weights << std::endl;
  output << spc.delta_threshold << std::endl;
  output << spc.stale_species << std::endl;
  return output;
}
