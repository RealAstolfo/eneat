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
    std::string &model_name)
    : get_fitness(get_fitness) {
  this->model_name = std::move(model_name);
  this->p = std::make_unique<pool>(1, 1, 1, true);
  load_pool(this->model_name + "_pool");
  load_best(this->model_name + "_best");

  srand(time(NULL));
  this->p->max_fitness.store(
      std::numeric_limits<size_t>::max());
}

model::~model() {
  save_best();   // No-op if read_only_
  save_pool();   // No-op if read_only_
}

// Coroutine-based training job for a batch of genomes
ethreads::coro_task<std::pair<size_t, genome *>>
training_job_coro(std::vector<genome *>::iterator start,
                  std::vector<genome *>::iterator stop,
                  fitness_func_t fit_func) {
  genome *best_genome = *start;
  size_t best_fitness = 0;

  for (; start != stop; std::advance(start, 1)) {
    brain net;
    genome *const genome = *start;
    net = *genome;
    size_t fitness = co_await fit_func(net);
    genome->fitness.store(fitness);
    if (fitness > best_fitness) {
      best_fitness = fitness;
      best_genome = genome;
    }
  }

  co_return std::pair<size_t, genome *>(best_fitness, best_genome);
}

void model::train(std::size_t times) {
  // Use stored best fitness for comparison
  size_t best_fitness = this->get_best_fitness();
  while (times-- > 0) {
    std::vector<genome *> genomes;
    for (auto &specie : this->p->species)
      std::transform(std::begin(specie.genomes), std::end(specie.genomes),
                     std::back_inserter(genomes),
                     [](genome &g) -> genome * { return &g; });

    std::sort(std::begin(genomes), std::end(genomes),
              [](const genome *a, const genome *b) { return a < b; });

    auto last = std::unique(std::begin(genomes), std::end(genomes));
    genomes.erase(last, std::end(genomes));

    // Create coroutine tasks for parallel genome evaluation
    std::vector<ethreads::coro_task<std::pair<size_t, genome *>>> tasks;
    const std::size_t num_threads = std::thread::hardware_concurrency();
    const std::size_t total_items = genomes.size();

    if (total_items == 0) {
      this->p->new_generation();
      continue;
    }

    const std::size_t batch_size = (total_items + num_threads - 1) / num_threads;
    auto batch_start = std::begin(genomes);

    while (batch_start != std::end(genomes)) {
      auto batch_stop = batch_start;
      std::size_t remaining = std::distance(batch_start, std::end(genomes));
      std::advance(batch_stop, std::min(batch_size, remaining));
      tasks.push_back(training_job_coro(batch_start, batch_stop, this->get_fitness));
      batch_start = batch_stop;
    }

    // Start all tasks for parallel execution
    for (auto &task : tasks) {
      task.start();
    }

    // Collect results (tasks run in parallel, collection is sequential)
    for (auto &task : tasks) {
      auto [fitness, genome] = task.get();
      if (fitness > best_fitness) {
        best_fitness = fitness;
        brain net;
        net = *genome;
        this->set_best_brain(net, fitness);
      }
    }

    this->p->new_generation();
  }
}

ethreads::coro_task<void> model::train_async(std::size_t times) {
  // Use stored best fitness for comparison
  size_t best_fitness = this->get_best_fitness();

  while (times-- > 0) {
    std::vector<genome *> genomes;
    for (auto &specie : this->p->species)
      std::transform(std::begin(specie.genomes), std::end(specie.genomes),
                     std::back_inserter(genomes),
                     [](genome &g) -> genome * { return &g; });

    std::sort(std::begin(genomes), std::end(genomes),
              [](const genome *a, const genome *b) { return a < b; });

    auto last = std::unique(std::begin(genomes), std::end(genomes));
    genomes.erase(last, std::end(genomes));

    if (genomes.empty()) {
      co_await this->p->new_generation_async();
      continue;
    }

    // Create evaluation tasks
    std::vector<ethreads::coro_task<std::pair<size_t, genome *>>> tasks;
    const std::size_t num_threads = std::thread::hardware_concurrency();
    const std::size_t batch_size = (genomes.size() + num_threads - 1) / num_threads;

    auto batch_start = std::begin(genomes);
    while (batch_start != std::end(genomes)) {
      auto batch_stop = batch_start;
      std::size_t remaining = std::distance(batch_start, std::end(genomes));
      std::advance(batch_stop, std::min(batch_size, remaining));
      tasks.push_back(training_job_coro(batch_start, batch_stop, this->get_fitness));
      batch_start = batch_stop;
    }

    // Use when_all for parallel evaluation
    auto results = co_await ethreads::when_all(std::move(tasks));

    for (auto &[fitness, genome] : results) {
      if (fitness > best_fitness) {
        best_fitness = fitness;
        brain net;
        net = *genome;
        this->set_best_brain(net, fitness);
      }
    }

    co_await this->p->new_generation_async();
  }
}

bool model::save_best() {
  if (read_only_) return false;  // Don't save in read-only mode
  const std::string name = this->model_name + "_best";
  std::ofstream of;
  of.open(name.data(), std::ios::trunc);
  zstream compressor(&of);
  compressor << best_state_.load().best;
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
    best_state_.store({loaded_best, 0});
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
