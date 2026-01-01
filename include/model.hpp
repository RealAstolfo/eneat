#ifndef ENEAT_MODEL_HPP
#define ENEAT_MODEL_HPP

#include <functional>
#include <memory>
#include <vector>

#include "brain.hpp"
#include "coro_task.hpp"
#include "pool.hpp"
#include "shared_state.hpp"

using fitness_func_t = std::function<ethreads::coro_task<size_t>(brain &)>;

// Combined state for best brain and its fitness
struct best_state_t {
  brain best;
  size_t fitness{0};
};

struct model {
  model(const fitness_func_t &get_fitness, std::string &model_name);

  ~model();

  void train(std::size_t times = 1);
  ethreads::coro_task<void> train_async(std::size_t times = 1);

  bool save_best();
  bool save_pool();
  bool load_best(std::string file_name);
  bool load_pool(std::string file_name);

  // Disable auto-save on destruction (for read-only/play mode)
  void set_read_only(bool read_only = true) { read_only_ = read_only; }
  bool is_read_only() const { return read_only_; }

  // Thread-safe access to best brain
  brain get_best_brain() const {
    return best_state_.load().best;
  }

  size_t get_best_fitness() const {
    return best_state_.load().fitness;
  }

  void set_best_brain(const brain& b, size_t fitness) {
    best_state_.store({b, fitness});
  }

  const fitness_func_t get_fitness;
  std::string model_name;
  ethreads::sync_shared_value<best_state_t> best_state_{};
  std::unique_ptr<pool> p;

private:
  bool read_only_ = false;
};

#endif
