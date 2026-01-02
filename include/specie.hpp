#ifndef ENEAT_SPECIE_HPP
#define ENEAT_SPECIE_HPP

#include <cstddef>
#include <vector>

#include "genome.hpp"
#include "shared_state.hpp"

struct specie {
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> top_fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> average_fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> staleness{0};
  // Thread-safe genomes vector using ethreads sync_shared_value with shared_mutex for reader-writer locking
  ethreads::sync_shared_value<std::vector<genome>, ethreads::shared_mutex_lock_policy> genomes{};

  specie() = default;

  // Copy constructor - sync_shared_value is not copyable, so we load and store
  specie(const specie &other)
      : top_fitness(other.top_fitness.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()),
        genomes(other.genomes.load()) {}

  // Copy assignment
  specie &operator=(const specie &other) {
    if (this != &other) {
      top_fitness.store(other.top_fitness.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      genomes.store(other.genomes.load());
    }
    return *this;
  }

  // Move constructor
  specie(specie &&other) noexcept
      : top_fitness(other.top_fitness.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()),
        genomes(other.genomes.load()) {}

  // Move assignment
  specie &operator=(specie &&other) noexcept {
    if (this != &other) {
      top_fitness.store(other.top_fitness.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      genomes.store(other.genomes.load());
    }
    return *this;
  }
};

#endif
