#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <thread>

#include "async_runtime.hpp"
#include "coro_task.hpp"
#include "genome.hpp"
#include "model.hpp"
#include "pool.hpp"
#include "specie.hpp"
#include "task_scheduler.hpp"
#include "utils.hpp"
#include "zstream.hpp"

model::model(
    const fitness_func_t &get_fitness,
    std::string &model_name,
    size_t input,
    size_t output,
    size_t population,
    size_t bias,
    bool recurrent)
    : get_fitness(get_fitness) {
  this->model_name = std::move(model_name);
  this->p = std::make_unique<pool>(input, output, population, bias, recurrent);
  load_pool(this->model_name + "_pool");
  load_best(this->model_name + "_best");

  srand(time(NULL));
}

model::~model() {
  save_best();   // No-op if read_only_
  save_pool();   // No-op if read_only_
}

// === rtNEAT: Real-time continuous evolution ===

size_t model::population_size() const {
  return p->get_population_size();
}

// Evaluate a genome copy and return fitness (safe across co_await points)
ethreads::coro_task<size_t> model::evaluate_genome_copy_async(genome g) {
  brain net;
  net = g;
  size_t fitness = co_await get_fitness(net);

  // Update best if this is better (thread-safe via sync_shared_value)
  if (fitness > get_best_fitness()) {
    set_best_brain(net, fitness);
  }

  co_return fitness;
}

// Single tick of rtNEAT evolution with safe parallel evaluation
ethreads::coro_task<void> model::tick_async() {
  // Get genome indices (safe - indices remain valid across co_await)
  auto indices = p->get_genome_indices();
  if (indices.empty()) {
    co_return;
  }

  // Determine batch size for this tick
  size_t pop_size = indices.size();
  size_t batch = std::min(batch_size_, pop_size);

  // Prepare batch evaluation
  eval_index_ = eval_index_ % pop_size;

  // Collect genome copies and create tasks
  std::vector<ethreads::coro_task<size_t>> tasks;
  std::vector<std::pair<size_t, size_t>> batch_indices;
  tasks.reserve(batch);
  batch_indices.reserve(batch);

  for (size_t i = 0; i < batch; i++) {
    size_t idx = (eval_index_ + i) % pop_size;
    auto [species_idx, genome_idx] = indices[idx];
    batch_indices.push_back({species_idx, genome_idx});

    // Get a COPY of the genome (safe to use across co_await)
    genome g = p->get_genome_copy(species_idx, genome_idx);
    tasks.push_back(evaluate_genome_copy_async(std::move(g)));
  }

  // Start all evaluation tasks in parallel
  for (auto& task : tasks) {
    task.start();
  }

  // Wait for all evaluations and update original genomes
  for (size_t i = 0; i < batch; i++) {
    size_t fitness = co_await tasks[i];
    auto [species_idx, genome_idx] = batch_indices[i];

    // Update original genome's fitness and age (thread-safe via pool methods)
    p->set_genome_fitness(species_idx, genome_idx, fitness);
    p->increment_genome_age(species_idx, genome_idx);
  }

  eval_index_ = (eval_index_ + batch) % pop_size;

  // === rtNEAT FEATURE INTEGRATION ===
  // Update running averages and perform rtNEAT operations periodically
  if (tick_count.load() % 10 == 0) {
    p->estimate_all_averages();

    // Calculate adjusted fitness for all organisms (CRITICAL - enables proper selection)
    // Without this, remove_worst() selects randomly since adjusted_fitness would be 0
    for (auto& s : p->species) {
      p->adjust_species_fitness(s);
    }

    // Calculate expected offspring proportionally based on species fitness
    p->calculate_expected_offspring();

    // Redistribute offspring from weak to strong species (babies stolen)
    p->redistribute_offspring();

    // Check for population stagnation and apply delta-coding if needed
    p->check_delta_coding();

    // Adjust compatibility threshold to maintain target species count
    p->adjust_compatibility_threshold();
  }

  // Population management: handle replacement (sync to avoid scheduler overhead)
  size_t current_pop = population_size();
  size_t target_pop = p->speciating_parameters.population;

  if (current_pop >= target_pop) {
    // Try to remove worst mature organism
    bool removed = p->remove_worst();

    if (removed) {
      // Only add offspring if we successfully removed one
      genome child = p->reproduce_one();
      p->add_to_species(std::move(child));
    }
    // If removal failed (no mature organisms), skip adding to let population age
  } else if (current_pop < target_pop) {
    // Population below target - just add offspring without removing
    genome child = p->reproduce_one();
    p->add_to_species(std::move(child));
  }

  // Per-generation operations (once per population_size ticks)
  // A "generation" in rtNEAT is approximately population_size ticks
  if (tick_count.load() % p->speciating_parameters.population == 0) {
    // Age all species (increment age counter)
    for (auto& s : p->species) {
      s.increment_age();
    }

    // Reset champion flags for new generation
    p->reset_champion_flags();
  }

  // Periodic species cleanup (every 10 generations)
  // Skip tick 0 to avoid removing all species before fitness is built
  size_t cleanup_period = p->speciating_parameters.population * 10;
  if (tick_count.load() > 0 && tick_count.load() % cleanup_period == 0) {
    p->remove_stale_species();
    p->remove_weak_species();
  }

  // Increment tick counter
  tick_count.store(tick_count.load() + 1);

  co_return;
}

// Run continuous evolution for N ticks
ethreads::coro_task<void> model::evolve_async(std::size_t ticks) {
  while (ticks-- > 0) {
    co_await tick_async();
  }
}

bool model::save_best() {
  if (read_only_) return false;  // Don't save in read-only mode
  const std::string name = this->model_name + "_best";
  std::ofstream of;
  of.open(name.data(), std::ios::trunc);
  zstream compressor(&of);
  brain best_to_save = get_best_brain();
  compressor << best_to_save;
  compressor << std::flush;
  of.close();
  return true;
}

bool model::save_pool() {
  if (read_only_) return false;  // Don't save in read-only mode
  const std::string name = this->model_name + "_pool";
  std::ofstream of;
  of.open(name.data(), std::ios::trunc);
  zstream compressor(&of);
  compressor << *this->p;
  compressor << std::flush;
  of.close();
  return true;
}

bool model::load_best(std::string file_name) {
  std::ifstream best_file;
  best_file.open(file_name.data());
  if (best_file.is_open()) {
    zstream decompressor(&best_file);
    brain loaded_best;
    decompressor >> loaded_best;
    // Use lock directly since we're loading (not comparing fitness)
    {
      std::lock_guard lock(best_brain_mutex_);
      best_brain_ = loaded_best;
    }
    best_fitness_.store(0, std::memory_order_release);
    return true;
  }

  return false;
}

bool model::load_pool(std::string file_name) {
  std::ifstream pool;
  pool.open(file_name.data());
  if (pool.is_open()) {
    zstream decompressor(&pool);
    decompressor >> *this->p;
    return true;
  }

  return false;
}
