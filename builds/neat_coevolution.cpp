// neat_coevolution.cpp - Competitive Coevolution & rtNEAT Control Demo
//
// This example demonstrates:
// - Champion/coevolution tracking: pop_champ, winner, eliminate, super_champ_offspring, error
// - Probabilistic selection: remove_worst_probabilistic()
// - rtNEAT control: age_all_organisms(), calculate_expected_offspring(), redistribute_offspring(),
//                   check_delta_coding(), apply_delta_coding()
// - Speciation: disjoint(), excess(), mut_diff(), dropoff_age, age_significance, reassign_all_species()
// - Breeding: breed_child_sync()
// - Analysis: calculate_network_depth(), verify_genome()

#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "coro_task.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>
#include <memory>

// Predator-Prey Arena
// Two agents compete: predator tries to catch prey, prey tries to escape.
// Simple 2D continuous space with wrapping boundaries.
struct Arena {
  struct Agent {
    float x, y;       // Position
    float vx, vy;     // Velocity
    float heading;    // Direction in radians

    Agent() : x(0), y(0), vx(0), vy(0), heading(0) {}

    void reset(float start_x, float start_y) {
      x = start_x;
      y = start_y;
      vx = vy = 0;
      heading = 0;
    }

    // Move based on turn and thrust inputs
    void move(float turn, float thrust) {
      // Update heading
      heading += turn * 0.2f;  // Max turn rate

      // Apply thrust
      float speed = thrust * 0.05f;  // Max speed
      vx = std::cos(heading) * speed;
      vy = std::sin(heading) * speed;

      // Update position with wrapping
      x += vx;
      y += vy;
      if (x < 0) x += 1.0f;
      if (x > 1.0f) x -= 1.0f;
      if (y < 0) y += 1.0f;
      if (y > 1.0f) y -= 1.0f;
    }
  };

  Agent predator;
  Agent prey;
  int steps;
  static constexpr int max_steps = 100;
  static constexpr float catch_distance = 0.05f;

  void reset() {
    // Random starting positions
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    predator.reset(dist(rng), dist(rng));
    prey.reset(dist(rng), dist(rng));

    // Ensure they start far enough apart
    while (distance() < 0.2f) {
      prey.reset(dist(rng), dist(rng));
    }

    steps = 0;
  }

  float distance() const {
    float dx = predator.x - prey.x;
    float dy = predator.y - prey.y;
    // Handle wrap-around distance
    if (dx > 0.5f) dx -= 1.0f;
    if (dx < -0.5f) dx += 1.0f;
    if (dy > 0.5f) dy -= 1.0f;
    if (dy < -0.5f) dy += 1.0f;
    return std::sqrt(dx*dx + dy*dy);
  }

  bool is_caught() const {
    return distance() < catch_distance;
  }

  bool is_done() const {
    return steps >= max_steps || is_caught();
  }

  // Get predator's view: [rel_x, rel_y, prey_heading, own_heading]
  std::vector<exfloat> get_predator_view() const {
    float dx = prey.x - predator.x;
    float dy = prey.y - predator.y;
    // Handle wrap-around
    if (dx > 0.5f) dx -= 1.0f;
    if (dx < -0.5f) dx += 1.0f;
    if (dy > 0.5f) dy -= 1.0f;
    if (dy < -0.5f) dy += 1.0f;

    return {
      static_cast<exfloat>(dx),
      static_cast<exfloat>(dy),
      static_cast<exfloat>(std::sin(prey.heading)),
      static_cast<exfloat>(std::cos(predator.heading))
    };
  }

  // Get prey's view: [rel_x, rel_y, predator_heading, own_heading]
  std::vector<exfloat> get_prey_view() const {
    float dx = predator.x - prey.x;
    float dy = predator.y - prey.y;
    if (dx > 0.5f) dx -= 1.0f;
    if (dx < -0.5f) dx += 1.0f;
    if (dy > 0.5f) dy -= 1.0f;
    if (dy < -0.5f) dy += 1.0f;

    return {
      static_cast<exfloat>(dx),
      static_cast<exfloat>(dy),
      static_cast<exfloat>(std::sin(predator.heading)),
      static_cast<exfloat>(std::cos(prey.heading))
    };
  }

  void step(float pred_turn, float pred_thrust, float prey_turn, float prey_thrust) {
    predator.move(pred_turn, pred_thrust);
    prey.move(prey_turn, prey_thrust);
    steps++;
  }
};

int main() {
  std::cerr << "NEAT Coevolution Demo - Predator-Prey" << std::endl;
  std::cerr << "======================================" << std::endl;
  std::cerr << "Features: Competitive coevolution, manual rtNEAT control, probabilistic selection" << std::endl;
  std::cerr << std::endl;

  // === Create two separate populations ===
  // Using pool directly for manual control (no model wrapper)

  std::cerr << "Initializing predator and prey populations..." << std::endl;

  // 4 inputs, 2 outputs (turn, thrust), 100 population each
  pool predators(4, 2, 100, 1, true);
  pool prey(4, 2, 100, 1, true);

  // Configure speciation parameters
  auto configure_pool = [](pool& p) {
    p.speciating_parameters.population = 100;
    p.speciating_parameters.target_species_count = 8;
    p.speciating_parameters.dynamic_threshold_enabled = true;
    p.speciating_parameters.time_alive_minimum = 3;

    // rtNEAT specific parameters
    p.speciating_parameters.dropoff_age = 50;        // Species age limit
    p.speciating_parameters.age_significance = 1.5f; // Youth bonus

    // Enable delta coding for stagnation recovery
    p.speciating_parameters.delta_coding_enabled = true;
    p.speciating_parameters.babies_stolen = 3;
  };

  configure_pool(predators);
  configure_pool(prey);

  std::cerr << "Predator population: " << predators.get_population_size() << std::endl;
  std::cerr << "Prey population: " << prey.get_population_size() << std::endl;
  std::cerr << std::endl;

  // === Demonstrate compatibility distance functions ===
  std::cerr << "=== Compatibility Distance Demo ===" << std::endl;
  {
    auto pred_indices = predators.get_genome_indices();
    if (pred_indices.size() >= 2) {
      auto g1 = predators.get_genome_copy(pred_indices[0].first, pred_indices[0].second);
      auto g2 = predators.get_genome_copy(pred_indices[1].first, pred_indices[1].second);

      float disjoint = predators.disjoint(g1, g2);
      float excess = predators.excess(g1, g2);
      float mut_diff = predators.mut_diff(g1, g2);
      bool same_species = predators.is_same_species(g1, g2);

      std::cerr << "Comparing two predator genomes:" << std::endl;
      std::cerr << "  Disjoint: " << disjoint << std::endl;
      std::cerr << "  Excess: " << excess << std::endl;
      std::cerr << "  Mutation diff: " << mut_diff << std::endl;
      std::cerr << "  Same species: " << (same_species ? "yes" : "no") << std::endl;
    }
  }
  std::cerr << std::endl;

  // === Main coevolution loop ===
  std::cerr << "Starting coevolution..." << std::endl;
  std::cerr << std::endl;

  Arena arena;
  size_t generation = 0;
  const size_t max_generations = 200;
  const size_t ticks_per_generation = 50;

  // Track evolution statistics
  size_t predator_wins = 0;
  size_t prey_wins = 0;
  size_t total_matches = 0;

  while (generation < max_generations) {
    // === rtNEAT: Age all organisms ===
    predators.age_all_organisms();
    prey.age_all_organisms();

    // === Competitive evaluation ===
    // Each predator competes against prey champions

    // Get prey champion for predator evaluation
    genome prey_champion;
    if (!prey.species.empty()) {
      prey_champion = prey.get_species_champion(prey.species.front());
    }

    // Get predator champion for prey evaluation
    genome pred_champion;
    if (!predators.species.empty()) {
      pred_champion = predators.get_species_champion(predators.species.front());
    }

    // Evaluate predators against prey champion
    auto pred_indices = predators.get_genome_indices();
    for (const auto& [sp_idx, g_idx] : pred_indices) {
      genome pred_genome = predators.get_genome_copy(sp_idx, g_idx);

      // Build brains
      brain pred_brain;
      pred_brain = pred_genome;

      brain prey_brain;
      prey_brain = prey_champion;

      // Run matches
      int wins = 0;
      constexpr int num_matches = 5;

      for (int m = 0; m < num_matches; m++) {
        arena.reset();

        while (!arena.is_done()) {
          // Predator action
          std::vector<exfloat> pred_out(2, 0.0f);
          pred_brain.evaluate(arena.get_predator_view(), pred_out);

          // Prey action
          std::vector<exfloat> prey_out(2, 0.0f);
          prey_brain.evaluate(arena.get_prey_view(), prey_out);

          arena.step(
            static_cast<float>(pred_out[0]),
            static_cast<float>(pred_out[1]),
            static_cast<float>(prey_out[0]),
            static_cast<float>(prey_out[1])
          );
        }

        if (arena.is_caught()) {
          wins++;
        }
      }

      // Set fitness and coevolution metadata
      size_t fitness = wins * 1000 + (num_matches - arena.steps / 20);
      predators.set_genome_fitness(sp_idx, g_idx, fitness);

      // Update coevolution tracking fields
      pred_genome.winner = (wins > num_matches / 2);
      pred_genome.error = static_cast<float>(num_matches - wins) / num_matches;
    }

    // Evaluate prey against predator champion
    auto prey_indices = prey.get_genome_indices();
    for (const auto& [sp_idx, g_idx] : prey_indices) {
      genome prey_genome = prey.get_genome_copy(sp_idx, g_idx);

      brain prey_brain;
      prey_brain = prey_genome;

      brain pred_brain;
      pred_brain = pred_champion;

      int escapes = 0;
      constexpr int num_matches = 5;

      for (int m = 0; m < num_matches; m++) {
        arena.reset();

        while (!arena.is_done()) {
          std::vector<exfloat> pred_out(2, 0.0f);
          pred_brain.evaluate(arena.get_predator_view(), pred_out);

          std::vector<exfloat> prey_out(2, 0.0f);
          prey_brain.evaluate(arena.get_prey_view(), prey_out);

          arena.step(
            static_cast<float>(pred_out[0]),
            static_cast<float>(pred_out[1]),
            static_cast<float>(prey_out[0]),
            static_cast<float>(prey_out[1])
          );
        }

        if (!arena.is_caught()) {
          escapes++;
        }
      }

      size_t fitness = escapes * 1000 + arena.steps;
      prey.set_genome_fitness(sp_idx, g_idx, fitness);

      prey_genome.winner = (escapes > num_matches / 2);
      prey_genome.error = static_cast<float>(num_matches - escapes) / num_matches;
    }

    // === rtNEAT: Calculate expected offspring ===
    predators.estimate_all_averages();
    prey.estimate_all_averages();

    predators.calculate_expected_offspring();
    prey.calculate_expected_offspring();

    // === rtNEAT: Redistribute offspring (babies stolen) ===
    predators.redistribute_offspring();
    prey.redistribute_offspring();

    // === rtNEAT: Check for stagnation ===
    predators.check_delta_coding();
    prey.check_delta_coding();

    // === Mark population champions ===
    auto mark_champion = [](pool& p) {
      auto best = p.get_best_genome();
      if (best) {
        best->pop_champ = true;
        best->super_champ_offspring = 5;  // Reserved offspring slots
      }
    };

    mark_champion(predators);
    mark_champion(prey);

    // === Reproduction using rtNEAT control ===
    for (size_t tick = 0; tick < ticks_per_generation; tick++) {
      // === Probabilistic removal (instead of worst-only) ===
      if (predators.get_population_size() >= predators.speciating_parameters.population) {
        predators.remove_worst_probabilistic();
      }
      if (prey.get_population_size() >= prey.speciating_parameters.population) {
        prey.remove_worst_probabilistic();
      }

      // === Breed new offspring using synchronous breeding ===
      if (!predators.species.empty()) {
        genome pred_child = predators.breed_child_sync(predators.species.front());

        // === Verify genome before adding ===
        if (predators.verify_genome(pred_child)) {
          // Track origin
          pred_child.generation_born = generation;

          // === Calculate network depth ===
          size_t depth = predators.calculate_network_depth(pred_child);
          if (depth > 10) {
            // Very deep network - might want to limit
          }

          predators.add_to_species(pred_child);
        }
      }

      if (!prey.species.empty()) {
        genome prey_child = prey.breed_child_sync(prey.species.front());

        if (prey.verify_genome(prey_child)) {
          prey_child.generation_born = generation;
          prey.add_to_species(prey_child);
        }
      }
    }

    // === rtNEAT: Adjust compatibility threshold ===
    predators.adjust_compatibility_threshold();
    prey.adjust_compatibility_threshold();

    // === Periodically reassign all species ===
    if (generation % 20 == 0 && generation > 0) {
      std::cerr << "[Gen " << generation << "] Reassigning all species..." << std::endl;
      predators.reassign_all_species();
      prey.reassign_all_species();
    }

    // === Reset champion flags for next generation ===
    predators.reset_champion_flags();
    prey.reset_champion_flags();

    // === Progress report ===
    if (generation % 10 == 0) {
      // Run a championship match
      brain pred_brain, prey_brain;

      auto pred_best = predators.get_best_genome();
      auto prey_best = prey.get_best_genome();

      if (pred_best && prey_best) {
        pred_brain = *pred_best;
        prey_brain = *prey_best;

        int champ_wins = 0;
        for (int m = 0; m < 10; m++) {
          arena.reset();
          while (!arena.is_done()) {
            std::vector<exfloat> pred_out(2, 0.0f);
            pred_brain.evaluate(arena.get_predator_view(), pred_out);
            std::vector<exfloat> prey_out(2, 0.0f);
            prey_brain.evaluate(arena.get_prey_view(), prey_out);
            arena.step(
              static_cast<float>(pred_out[0]),
              static_cast<float>(pred_out[1]),
              static_cast<float>(prey_out[0]),
              static_cast<float>(prey_out[1])
            );
          }
          if (arena.is_caught()) champ_wins++;
        }

        if (champ_wins > 5) predator_wins++;
        else prey_wins++;
        total_matches++;

        std::cerr << "[Gen " << generation << "] "
                  << "Pred pop: " << predators.get_population_size()
                  << "/" << predators.speciating_parameters.population
                  << " (" << predators.species.size() << " sp) | "
                  << "Prey pop: " << prey.get_population_size()
                  << "/" << prey.speciating_parameters.population
                  << " (" << prey.species.size() << " sp) | "
                  << "Champ match: " << champ_wins << "/10 catches"
                  << std::endl;

        // Show network depths
        size_t pred_depth = predators.calculate_network_depth(*pred_best);
        size_t prey_depth = prey.calculate_network_depth(*prey_best);
        std::cerr << "           Network depth - Pred: " << pred_depth
                  << ", Prey: " << prey_depth << std::endl;
      }
    }

    generation++;
  }

  // === Final results ===
  std::cerr << std::endl;
  std::cerr << "=== Coevolution Complete ===" << std::endl;
  std::cerr << "Generations: " << generation << std::endl;
  std::cerr << "Championship results: Predators " << predator_wins
            << " - " << prey_wins << " Prey" << std::endl;
  std::cerr << std::endl;

  // === Display best genomes' coevolution metadata ===
  std::cerr << "=== Best Predator Genome ===" << std::endl;
  auto pred_best = predators.get_best_genome();
  if (pred_best) {
    std::cerr << "Fitness: " << pred_best->fitness.load() << std::endl;
    std::cerr << "Generation born: " << pred_best->generation_born << std::endl;
    std::cerr << "Time alive: " << pred_best->time_alive.load() << std::endl;
    std::cerr << "Winner: " << (pred_best->winner ? "yes" : "no") << std::endl;
    std::cerr << "Error: " << pred_best->error << std::endl;
    std::cerr << "Pop champion: " << (pred_best->pop_champ ? "yes" : "no") << std::endl;
    std::cerr << "Super champ offspring: " << pred_best->super_champ_offspring << std::endl;
    std::cerr << "Active genes (extrons): " << pred_best->extrons() << "/" << pred_best->genes.size() << std::endl;

    brain b;
    b = *pred_best;
    std::cerr << "Neurons: " << b.neurons.size() << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "=== Best Prey Genome ===" << std::endl;
  auto prey_best = prey.get_best_genome();
  if (prey_best) {
    std::cerr << "Fitness: " << prey_best->fitness.load() << std::endl;
    std::cerr << "Generation born: " << prey_best->generation_born << std::endl;
    std::cerr << "Time alive: " << prey_best->time_alive.load() << std::endl;
    std::cerr << "Winner: " << (prey_best->winner ? "yes" : "no") << std::endl;
    std::cerr << "Error: " << prey_best->error << std::endl;
    std::cerr << "High fit: " << prey_best->high_fit << std::endl;

    brain b;
    b = *prey_best;
    std::cerr << "Neurons: " << b.neurons.size() << std::endl;
  }

  // === Final demonstration match ===
  std::cerr << std::endl;
  std::cerr << "=== Final Demonstration Match ===" << std::endl;

  if (pred_best && prey_best) {
    brain pred_brain, prey_brain;
    pred_brain = *pred_best;
    prey_brain = *prey_best;

    arena.reset();
    std::cerr << "Initial: Pred(" << arena.predator.x << "," << arena.predator.y
              << ") Prey(" << arena.prey.x << "," << arena.prey.y << ")" << std::endl;

    while (!arena.is_done()) {
      std::vector<exfloat> pred_out(2, 0.0f);
      pred_brain.evaluate(arena.get_predator_view(), pred_out);
      std::vector<exfloat> prey_out(2, 0.0f);
      prey_brain.evaluate(arena.get_prey_view(), prey_out);

      arena.step(
        static_cast<float>(pred_out[0]),
        static_cast<float>(pred_out[1]),
        static_cast<float>(prey_out[0]),
        static_cast<float>(prey_out[1])
      );

      if (arena.steps % 20 == 0) {
        std::cerr << "Step " << arena.steps << ": Distance=" << arena.distance()
                  << " Pred(" << arena.predator.x << "," << arena.predator.y << ")"
                  << std::endl;
      }
    }

    std::cerr << "Result: " << (arena.is_caught() ? "PREDATOR WINS!" : "PREY ESCAPES!")
              << " (steps=" << arena.steps << ", dist=" << arena.distance() << ")"
              << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "Done!" << std::endl;

  return 0;
}
