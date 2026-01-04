#ifndef ENEAT_MODEL_HPP
#define ENEAT_MODEL_HPP

#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "brain.hpp"
#include "coro_task.hpp"
#include "pool.hpp"
#include "shared_state.hpp"
#include "task_scheduler.hpp"

using fitness_func_t = std::function<ethreads::coro_task<size_t>(brain &)>;

struct model {
  model(const fitness_func_t &get_fitness, std::string &model_name,
        size_t input, size_t output, size_t population = 150,
        size_t bias = 1, bool recurrent = false);
  ~model();

  // === rtNEAT-style decoupled evolution ===

  // Evaluate batch of genomes in parallel (no replacement - decoupled from evolution)
  // Use this for real-time evaluation where evolution timing is controlled separately
  ethreads::coro_task<void> evaluate_batch_async();

  // rtNEAT iteration step: remove -> estimate -> reproduce (call-based)
  // Returns true if an organism was replaced, false if no mature organisms to remove
  // Reference: nero_evolution.cpp evolveBrains() - key insight: removal happens FIRST
  bool iteration_step();

  // === Combined evaluation + evolution (convenience) ===

  // Single tick of evolution: evaluate batch in parallel, replace worst if at capacity
  ethreads::coro_task<void> tick_async();

  // Run continuous evolution for N ticks
  ethreads::coro_task<void> evolve_async(std::size_t ticks);

  // Evaluate a genome copy and return fitness (safe across co_await points)
  ethreads::coro_task<size_t> evaluate_genome_copy_async(genome g);

  // Set parallel evaluation batch size (default: hardware_concurrency)
  void set_batch_size(size_t size) { batch_size_ = size; }
  size_t get_batch_size() const { return batch_size_; }

  // Get current population size
  size_t population_size() const;

  // Persistence
  bool save_pool();
  bool load_pool(std::string file_name);

  // Disable auto-save on destruction (for read-only/play mode)
  void set_read_only(bool read_only = true) { read_only_ = read_only; }
  bool is_read_only() const { return read_only_; }

  const fitness_func_t get_fitness;
  std::string model_name;
  std::unique_ptr<pool> p;

  // Tick counter for continuous evolution
  ethreads::sync_shared_value<size_t> tick_count{0};

private:
  bool read_only_ = false;
  size_t eval_index_ = 0;  // Round-robin evaluation index
  size_t batch_size_ = std::thread::hardware_concurrency();  // Parallel eval batch size
};

#endif
