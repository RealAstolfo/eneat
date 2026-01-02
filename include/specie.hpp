#ifndef ENEAT_SPECIE_HPP
#define ENEAT_SPECIE_HPP

#include <cstddef>
#include <vector>

#include "genome.hpp"
#include "shared_state.hpp"

struct specie {
  size_t id = 0;                    // Unique species identifier
  size_t age = 0;                   // Generations since creation
  size_t age_of_last_improvement = 0; // For stagnation detection

  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> top_fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> max_fitness_ever{0};  // Historical max
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> average_fitness{0};
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> staleness{0};

  // rtNEAT: Running average estimate for mid-generation selection
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> average_est{0};

  // Expected offspring for this species (calculated during selection)
  ethreads::sync_shared_value<size_t, ethreads::mutex_lock_policy> expected_offspring{0};

  // Thread-safe genomes vector using ethreads sync_shared_value with shared_mutex for reader-writer locking
  ethreads::sync_shared_value<std::vector<genome>, ethreads::shared_mutex_lock_policy> genomes{};

  // Flag for newly created species (protects from aging in first gen)
  bool novel = true;

  specie() = default;

  specie(size_t species_id) : id(species_id) {}

  // Copy constructor - sync_shared_value is not copyable, so we load and store
  specie(const specie &other)
      : id(other.id), age(other.age),
        age_of_last_improvement(other.age_of_last_improvement),
        top_fitness(other.top_fitness.load()),
        max_fitness_ever(other.max_fitness_ever.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()),
        average_est(other.average_est.load()),
        expected_offspring(other.expected_offspring.load()),
        genomes(other.genomes.load()),
        novel(other.novel) {}

  // Copy assignment
  specie &operator=(const specie &other) {
    if (this != &other) {
      id = other.id;
      age = other.age;
      age_of_last_improvement = other.age_of_last_improvement;
      top_fitness.store(other.top_fitness.load());
      max_fitness_ever.store(other.max_fitness_ever.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      average_est.store(other.average_est.load());
      expected_offspring.store(other.expected_offspring.load());
      genomes.store(other.genomes.load());
      novel = other.novel;
    }
    return *this;
  }

  // Move constructor
  specie(specie &&other) noexcept
      : id(other.id), age(other.age),
        age_of_last_improvement(other.age_of_last_improvement),
        top_fitness(other.top_fitness.load()),
        max_fitness_ever(other.max_fitness_ever.load()),
        average_fitness(other.average_fitness.load()),
        staleness(other.staleness.load()),
        average_est(other.average_est.load()),
        expected_offspring(other.expected_offspring.load()),
        genomes(other.genomes.load()),
        novel(other.novel) {}

  // Move assignment
  specie &operator=(specie &&other) noexcept {
    if (this != &other) {
      id = other.id;
      age = other.age;
      age_of_last_improvement = other.age_of_last_improvement;
      top_fitness.store(other.top_fitness.load());
      max_fitness_ever.store(other.max_fitness_ever.load());
      average_fitness.store(other.average_fitness.load());
      staleness.store(other.staleness.load());
      average_est.store(other.average_est.load());
      expected_offspring.store(other.expected_offspring.load());
      genomes.store(other.genomes.load());
      novel = other.novel;
    }
    return *this;
  }

  // Increment species age
  void increment_age() {
    if (!novel) {
      age++;
    }
    novel = false;
  }

  // Update fitness tracking
  void update_fitness(size_t new_top) {
    if (new_top > max_fitness_ever.load()) {
      max_fitness_ever.store(new_top);
      age_of_last_improvement = age;
      staleness.store(0);
    }
  }
};

#endif
