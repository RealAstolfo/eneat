#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>

#include "genome.hpp"
#include "model.hpp"
#include "pool.hpp"
#include "specie.hpp"
#include "task_scheduler.hpp"
#include "utils.hpp"
#include "zstream.hpp"

model::model(
    const std::function<decltype(genome::fitness)(brain &)> &get_fitness,
    std::string &model_name)
    : get_fitness(get_fitness) {
  this->model_name = std::move(model_name);
  this->p = std::make_unique<pool>(1, 1, 1, true);
  load_pool(this->model_name + "_pool");
  load_best(this->model_name + "_best");

  srand(time(NULL));
  this->p->max_fitness =
      std::numeric_limits<decltype(this->p->max_fitness)>::max();
}

model::~model() {
  save_best();
  save_pool();
}

std::pair<decltype(genome::fitness), genome *>
training_job(std::vector<genome>::iterator start,
             std::vector<genome>::iterator stop,
             std::function<decltype(genome::fitness)(brain &)> fit_func) {
  genome &best_genome = *start;
  decltype(genome::fitness) best_fitness = 0;

  for (; start != stop; std::advance(start, 1)) {
    brain net;
    net = *start;
    start->fitness = fit_func(net);
    if (start->fitness > best_fitness) {
      best_fitness = start->fitness;
      best_genome = *start;
    }
  }

  return std::pair<decltype(best_fitness), genome *>(best_fitness,
                                                     &best_genome);
}

void model::train(std::size_t times) {
  decltype(genome::fitness) best_fitness = this->get_fitness(best);
  while (times-- > 0) {
    std::vector<std::future<std::pair<decltype(genome::fitness), genome *>>>
        futures;

    for (auto s = std::begin(this->p->species); s != std::end(this->p->species);
         std::advance(s, 1)) {

      auto lfutures = add_batch_task(training_job, std::begin(s->genomes),
                                     std::end(s->genomes), this->get_fitness);
      futures.insert(std::end(futures),
                     std::make_move_iterator(std::begin(lfutures)),
                     std::make_move_iterator(std::end(lfutures)));
    }

    for (auto &future : futures) {
      auto [fitness, genome] = future.get();
      if (fitness > best_fitness) {
        best_fitness = fitness;
        this->best = *genome;
      }
    }

    this->p->new_generation();
  }
}

bool model::save_best() {
  const std::string name = this->model_name + "_best";
  std::ofstream of;
  of.open(name.data(), std::ios::trunc);
  zstream compressor(&of);
  compressor << best;
  compressor << std::flush;
  of.close();
  return true;
}

bool model::save_pool() {
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
  std::ifstream best;
  best.open(file_name.data());
  if (best.is_open()) {
    zstream decompressor(&best);
    decompressor >> this->best;
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
