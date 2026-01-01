#ifndef ENEAT_SPECIE_HPP
#define ENEAT_SPECIE_HPP

#include <cstddef>
#include <mutex>
#include <vector>

#include "genome.hpp"
#include "shared_state.hpp"

struct specie {
  ethreads::sync_shared_value<size_t> top_fitness{0};
  ethreads::sync_shared_value<size_t> average_fitness{0};
  ethreads::sync_shared_value<size_t> staleness{0};
  std::vector<genome> genomes;
  // Mutex to protect genomes vector during concurrent access
  mutable std::mutex genomes_mutex;

  specie() = default;

  // Copy constructor - sync_shared_value and mutex are not copyable
  specie(const specie &other)
      : top_fitness(other.top_fitness.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()) {
    std::lock_guard<std::mutex> lock(other.genomes_mutex);
    genomes = other.genomes;
  }

  // Copy assignment
  specie &operator=(const specie &other) {
    if (this != &other) {
      // Lock both mutexes in a consistent order to avoid deadlock
      std::scoped_lock lock(genomes_mutex, other.genomes_mutex);
      top_fitness.store(other.top_fitness.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      genomes = other.genomes;
    }
    return *this;
  }

  // Move constructor
  specie(specie &&other) noexcept
      : top_fitness(other.top_fitness.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()) {
    // Note: Moving from other, so we need to lock it
    std::lock_guard<std::mutex> lock(other.genomes_mutex);
    genomes = std::move(other.genomes);
  }

  // Move assignment
  specie &operator=(specie &&other) noexcept {
    if (this != &other) {
      std::scoped_lock lock(genomes_mutex, other.genomes_mutex);
      top_fitness.store(other.top_fitness.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      genomes = std::move(other.genomes);
    }
    return *this;
  }
};

#endif
