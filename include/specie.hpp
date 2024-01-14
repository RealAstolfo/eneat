#ifndef SPECIE_HPP
#define SPECIE_HPP

#include <cstddef>
#include <vector>

#include "genome.hpp"

struct specie {
  size_t top_fitness = 0;
  size_t average_fitness = 0;
  size_t staleness = 0;
  std::vector<genome> genomes;
};

#endif
