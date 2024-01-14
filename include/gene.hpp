#ifndef GENE_HPP
#define GENE_HPP

#include "functions.hpp"
#include "math.hpp"

struct gene {
  size_t innovation_num = -1;
  size_t from_node = -1;
  size_t to_node = -1;
  exfloat weight = 0.0f;
  ai_func_type activation = ai_func_type::RELU;
  bool enabled = true;
};

#endif
