#ifndef ENEAT_MODEL_HPP
#define ENEAT_MODEL_HPP

#include <functional>
#include <memory>
#include <vector>

#include "brain.hpp"
#include "pool.hpp"

struct model {
  model(const std::function<decltype(genome::fitness)(brain &)> &get_fitness,
        std::string &model_name);

  ~model();

  void train(std::size_t times = 1);

  bool save_best();
  bool save_pool();
  bool load_best(std::string file_name);
  bool load_pool(std::string file_name);

  const std::function<decltype(genome::fitness)(brain &)> get_fitness;
  std::string model_name;
  brain best;
  std::unique_ptr<pool> p;
};

#endif
