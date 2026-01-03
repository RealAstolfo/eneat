#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "network_visualizer.hpp"
#include "shared_state.hpp"
#include <limits>

#define xor_max_n 4
#define needed_fitness 0.75

const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
    {{0.0, 0.0}, 0.0f},
    {{0.0, 1.0}, 1.0f},
    {{1.0, 0.0}, 1.0f},
    {{1.0, 1.0}, 0.0f}};

int main() {
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

  std::string model_name = "xor_coro";
  // model(fitness_func, name, inputs, outputs, population, bias, recurrent)
  model xor_model(fitness_function, model_name, 2, 1, 150, 1, true);

  // === Configure Dynamic Population Control Parameters ===
  auto& params = xor_model.p->speciating_parameters;
  params.population = 150;                    // Target population size
  params.time_alive_minimum = 5;              // Maturity age before selection eligibility
  params.delta_coding_enabled = true;         // Enable stagnation recovery
  params.babies_stolen = 5;                   // Offspring redistribution to strong species
  params.dynamic_threshold_enabled = true;    // Auto-adjust species count
  params.target_species_count = 10;           // Initial species target (will adapt)
  params.compat_adjust_frequency = 10;        // Adjust threshold every 10 offspring
  params.compat_threshold_delta = 0.1f;       // Threshold adjustment step
  params.survival_thresh = 0.2f;              // Top 20% survive each generation
  
  // Initialize visualizer on main thread
  NetworkVisualizer visualizer;
  visualizer.open();

  // Shared state for visualization
  ethreads::sync_shared_value<brain> best_brain_copy{};
  ethreads::manual_reset_event training_done{false};
  ethreads::auto_reset_event update_ready{false};

  // rtNEAT training coroutine - continuous evolution without generations
  auto training_task = [&]() -> ethreads::coro_task<void> {
    size_t current_best;
    size_t last_update_tick = 0;

    do {
      // Run 100 ticks of evolution per update
      co_await xor_model.evolve_async(100);

      current_best = xor_model.p->max_fitness.load();
      size_t current_tick = xor_model.tick_count.load();
      float current_fitness = current_best / (exfloat)std::numeric_limits<size_t>::max();

      // === Dynamic Population Control: Adaptive species targeting ===
      // Low fitness -> more species for exploration
      // High fitness -> fewer species for exploitation
      if (current_fitness < 0.3f) {
        params.target_species_count = 15;  // Explore: many species
        params.population = 200;            // Larger population for diversity
      } else if (current_fitness < 0.5f) {
        params.target_species_count = 10;  // Balanced
        params.population = 150;
      } else {
        params.target_species_count = 6;   // Exploit: fewer species
        params.population = 100;            // Focus on best solutions
      }

      // Update visualization every 10 ticks
      if (current_tick - last_update_tick >= 10) {
        last_update_tick = current_tick;

        // Copy best brain for visualization (thread-safe)
        auto best = xor_model.p->get_best_brain();
        if (best) {
          best_brain_copy.store(*best);
          update_ready.set();

          // Enhanced output with population dynamics
          const char* mode = current_fitness < 0.3f ? "Exploring" :
                            (current_fitness < 0.5f ? "Balanced" : "Exploiting");

          std::cerr
              << "[Tick " << current_tick << "] "
              << "Pop: " << xor_model.population_size() << "/" << params.population
              << " | Species: " << xor_model.p->species.size() << "/" << params.target_species_count
              << " | Fitness: " << current_fitness
              << " | Mode: " << mode
              << " | Neurons: " << best->neurons.size()
              << "        \r";
        }
      }
    } while (current_best /
                 (exfloat)std::numeric_limits<size_t>::max() <
             needed_fitness);

    std::cerr << std::endl;

    auto final_brain = xor_model.p->get_best_brain();
    if (final_brain) {
      std::vector<exfloat> output = {0};
      for (const auto &[test_input, expected_output] : test_cases) {
        final_brain->evaluate(test_input, output);
        std::cerr << test_input[0] << " " << test_input[1] << " -> "
                  << (int)(output[0] + 0.5f) << std::endl;
      }
    }

    std::cerr << "Training complete after " << xor_model.tick_count.load()
              << " ticks. Close visualization window to exit." << std::endl;
    training_done.set();
    co_return;
  };

  // Start training in background
  auto task = training_task();
  task.start();

  // Main thread render loop - keeps OpenGL context on main thread
  while (visualizer.is_open()) {
    // Check if update is ready to render
    if (update_ready.try_wait()) {
      visualizer.render(best_brain_copy.load());
    } else if (training_done.is_set()) {
      // Training done, keep rendering final state
      visualizer.render(best_brain_copy.load());
    }
    visualizer.update();
  }

  // Wait for training to complete if window closed early
  if (!training_done.is_set()) {
    task.get();
  }

  return 0;
}
