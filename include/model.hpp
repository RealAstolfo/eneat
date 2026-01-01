#ifndef ENEAT_MODEL_HPP
#define ENEAT_MODEL_HPP

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "brain.hpp"
#include "coro_task.hpp"
#include "pool.hpp"

using fitness_func_t = std::function<ethreads::coro_task<size_t>(brain &)>;

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
    std::lock_guard<std::mutex> lock(best_mutex_);
    return best;
  }

  void set_best_brain(const brain& b, size_t fitness) {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best = b;
    best_fitness.store(fitness);
  }

  const fitness_func_t get_fitness;
  std::string model_name;
  brain best;
  std::atomic<size_t> best_fitness{0};  // Tracks best brain's fitness
  std::unique_ptr<pool> p;
  mutable std::mutex best_mutex_;  // Protects 'best' brain access

private:
  bool read_only_ = false;
};

#endif
