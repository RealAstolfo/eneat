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
#include <time.h>

#define xor_max_n 4
#define needed_fitness 0.75

const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
    {{0.0, 0.0}, 0.0f},
    {{0.0, 1.0}, 1.0f},
    {{1.0, 0.0}, 1.0f},
    {{1.0, 1.0}, 0.0f}};

// Simple synchronous fitness function for testing
size_t evaluate_sync(const genome& g) {
  brain net;
  net = g;

  if (net.neurons.size() > xor_max_n)
    return 0;

  std::vector<exfloat> output(1, 0.0);
  exfloat fitness = 0.0f;

  for (const auto &[test_input, expected_output] : test_cases) {
    net.evaluate(test_input, output);
    fitness += std::lerp(1.0f, 0.0f,
                         std::min(1.0f, error(output[0], expected_output)));
  }

  return std::lerp(0.0f,
                   (exfloat)std::numeric_limits<size_t>::max(),
                   fitness / 4.0f);
}

int main() {
  // Still use the async fitness function type for model compatibility
  const auto fitness_function = [](brain &net) -> ethreads::coro_task<size_t> {
    if (net.neurons.size() > xor_max_n)
      co_return 0;

    std::vector<exfloat> output(1, 0.0);
    exfloat fitness = 0.0f;

    for (const auto &[test_input, expected_output] : test_cases) {
      net.evaluate(test_input, output);
      fitness += std::lerp(1.0f, 0.0f,
                           std::min(1.0f, error(output[0], expected_output)));
    }

    co_return std::lerp(0.0f,
                     (exfloat)std::numeric_limits<size_t>::max(),
                     fitness / 4.0f);
  };

  std::string model_name = "xor";
  // Start with small population (100) to demonstrate dynamic growth
  model xor_model(fitness_function, model_name, 2, 1, 100, 1, true);

  // === Configure Dynamic Population Control Parameters ===
  auto& params = xor_model.p->speciating_parameters;
  params.population = 100;                    // Initial target (will change dynamically)
  params.time_alive_minimum = 5;              // Maturity age before selection eligibility
  params.delta_coding_enabled = true;         // Enable stagnation recovery
  params.babies_stolen = 5;                   // Offspring redistribution to strong species
  params.dynamic_threshold_enabled = true;    // Auto-adjust species count
  params.target_species_count = 8;            // Desired number of species
  params.compat_adjust_frequency = 10;        // Adjust threshold every 10 offspring
  params.compat_threshold_delta = 0.1f;       // Threshold adjustment step
  params.survival_thresh = 0.2f;              // Top 20% survive each generation

  std::cerr << "Starting evolution with dynamic population control..." << std::endl;
  std::cerr << "Phase 1: Growing population (target: 100 -> 200)" << std::endl;

  // Use fully asynchronous evolution via model::tick_async()
  size_t last_report_tick = 0;
  size_t tick = 0;

  size_t current_phase = 1;
  float last_threshold = params.delta_threshold;

  while (xor_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max() < needed_fitness) {
    // === Dynamic Population Control: Phase-based target adjustment ===
    if (tick >= 5000 && current_phase == 1) {
      current_phase = 2;
      params.population = 200;  // Grow population
      params.target_species_count = 12;  // More species for exploration
      std::cerr << "\n[Phase 2] Growing population: target 200, species target 12" << std::endl;
    } else if (tick >= 15000 && current_phase == 2) {
      current_phase = 3;
      params.population = 150;  // Stabilize population
      params.target_species_count = 8;   // Fewer species for exploitation
      std::cerr << "\n[Phase 3] Stabilizing: target 150, species target 8" << std::endl;
    }

    // Use tick_async for async evaluation
    auto tick_task = xor_model.tick_async();
    tick_task.start();
    tick_task.get();  // Wait for tick to complete

    // Enhanced progress report with population dynamics
    if (tick - last_report_tick >= 1000) {
      last_report_tick = tick;
      float fitness = xor_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max();
      float threshold_change = params.delta_threshold - last_threshold;
      last_threshold = params.delta_threshold;

      std::cerr << "[Tick " << tick << "] "
                << "Pop: " << xor_model.population_size() << "/" << params.population
                << " | Species: " << xor_model.p->species.size() << "/" << params.target_species_count
                << " | Fitness: " << fitness
                << " | Thresh: " << params.delta_threshold;

      if (threshold_change > 0.01f) {
        std::cerr << " (+)";
      } else if (threshold_change < -0.01f) {
        std::cerr << " (-)";
      }

      std::cerr << std::endl;
      std::cerr << "  └─ Delta coding: " << (params.delta_coding_enabled ? "ON" : "OFF")
                << " | Babies stolen: " << params.babies_stolen
                << " | Phase: " << current_phase
                << std::endl;
    }
    tick++;

    // Safety limit
    if (tick > 1000000) {
      std::cerr << "Reached tick limit" << std::endl;
      break;
    }
  }

  // Test coroutine infrastructure separately
  std::cerr << "\nTesting coroutine infrastructure..." << std::endl;

  // Test 1: Simple coroutine
  std::cerr << "Test 1: Simple coroutine..." << std::flush;
  auto simple_task = []() -> ethreads::coro_task<int> {
    co_return 42;
  };

  auto task = simple_task();
  task.start();
  int result = task.get();
  std::cerr << " result=" << result << " OK" << std::endl;

  // Test 2: Single fitness function coroutine
  std::cerr << "Test 2: Single fitness evaluation..." << std::flush;
  {
    genome g = xor_model.p->get_genome_copy(0, 0);
    brain net;
    net = g;
    auto fitness_task = fitness_function(net);
    fitness_task.start();
    size_t fitness = fitness_task.get();
    std::cerr << " fitness=" << fitness << " OK" << std::endl;
  }

  // Test 3: Sequential fitness evaluations
  std::cerr << "Test 3: Sequential fitness evaluations (5x)..." << std::flush;
  {
    for (int i = 0; i < 5; i++) {
      genome g = xor_model.p->get_genome_copy(0, 0);
      brain net;
      net = g;
      auto fitness_task = fitness_function(net);
      fitness_task.start();
      size_t fitness = fitness_task.get();
      (void)fitness;
    }
    std::cerr << " OK" << std::endl;
  }

  // Test 4: Parallel tasks (the pattern tick_async uses)
  std::cerr << "Test 4: Parallel fitness evaluations (5x)..." << std::flush;
  {
    std::vector<ethreads::coro_task<size_t>> tasks;
    std::vector<brain> brains;  // Keep brains alive during evaluation

    for (int i = 0; i < 5; i++) {
      genome g = xor_model.p->get_genome_copy(0, 0);
      brain net;
      net = g;
      brains.push_back(net);
    }

    // Create all tasks
    for (auto& net : brains) {
      tasks.push_back(fitness_function(net));
    }

    // Start all tasks in parallel
    for (auto& t : tasks) {
      t.start();
    }

    // Wait for all sequentially
    for (auto& t : tasks) {
      size_t fitness = t.get();
      (void)fitness;
    }
    std::cerr << " OK" << std::endl;
  }

  // Test 5: Nested coroutine (like evaluate_genome_copy_async)
  std::cerr << "Test 5: Nested coroutine..." << std::flush;
  {
    auto inner_coro = [&fitness_function](brain& net) -> ethreads::coro_task<size_t> {
      size_t fitness = co_await fitness_function(net);
      co_return fitness;
    };

    genome g = xor_model.p->get_genome_copy(0, 0);
    brain net;
    net = g;
    auto nested_task = inner_coro(net);
    nested_task.start();
    size_t fitness = nested_task.get();
    std::cerr << " fitness=" << fitness << " OK" << std::endl;
  }

  // Test 6: Parallel nested coroutines (exactly what tick_async does)
  std::cerr << "Test 6: Parallel nested coroutines (5x)..." << std::flush;
  {
    auto inner_coro = [&fitness_function](brain net) -> ethreads::coro_task<size_t> {
      size_t fitness = co_await fitness_function(net);
      co_return fitness;
    };

    std::vector<ethreads::coro_task<size_t>> tasks;

    for (int i = 0; i < 5; i++) {
      genome g = xor_model.p->get_genome_copy(0, 0);
      brain net;
      net = g;
      tasks.push_back(inner_coro(net));
    }

    // Start all
    for (auto& t : tasks) {
      t.start();
    }

    // Wait all
    for (auto& t : tasks) {
      size_t fitness = t.get();
      (void)fitness;
    }
    std::cerr << " OK" << std::endl;
  }

  std::cerr << "All coroutine tests passed!" << std::endl;

  // Display results
  std::cerr << std::endl;
  std::cerr << "Completed in " << tick << " ticks" << std::endl;
  std::cerr << "Final fitness: " << xor_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max() << std::endl;

  auto best_brain = xor_model.p->get_best_brain();
  if (best_brain) {
    std::vector<exfloat> output = {0};
    std::cerr << "XOR results:" << std::endl;
    for (const auto &[test_input, expected_output] : test_cases) {
      best_brain->evaluate(test_input, output);
      std::cerr << test_input[0] << " XOR " << test_input[1] << " -> "
                << output[0] << " (expected: " << expected_output << ")"
                << std::endl;
    }
  } else {
    std::cerr << "No best brain available (no evolution occurred)" << std::endl;
  }

  return 0;
}
