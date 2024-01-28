#ifndef ENEAT_NETWORK_INFO_CONTAINER_HPP
#define ENEAT_NETWORK_INFO_CONTAINER_HPP

#include <iostream>
#include <stdbool.h>
#include <stdlib.h>

struct network_info_container {
  size_t input_size;
  size_t bias_size;
  size_t output_size;
  size_t functional_neurons;
  bool recurrent;
};

#endif
