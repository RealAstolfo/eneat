#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "network_visualizer.hpp"
#include <limits>
#include <atomic>
#include <mutex>

#define xor_max_n 4
#define needed_fitness 0.95

const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
    {{0.0, 0.0}, 0.0f},
    {{0.0, 1.0}, 1.0f},
    {{1.0, 0.0}, 1.0f},
    {{1.0, 1.0}, 0.0f}};

int main() {
  const auto fitness_function = [](brain &net) -> decltype(genome::fitness) {
    if (net.neurons.size() > xor_max_n)
      return 0.0f;

    std::vector<exfloat> output(1, 0.0);
    exfloat fitness = 0.0f;

    for (const auto &[test_input, expected_output] : test_cases) {
      net.evaluate(test_input, output);
      fitness += std::lerp(1.0f, 0.0f,
                           std::min(1.0f, error(output[0], expected_output)));
    }

    return std::lerp(0.0f,
                     std::numeric_limits<decltype(genome::fitness)>::max(),
                     fitness / 4.0f);
  };

  std::string model_name = "xor_coro";
  model xor_model(fitness_function, model_name);
  xor_model.p->speciating_parameters.population = 50;
  xor_model.p->speciating_parameters.stale_species = std::numeric_limits<
      decltype(xor_model.p->speciating_parameters.stale_species)>::max();

  // Initialize visualizer on main thread
  NetworkVisualizer visualizer;
  visualizer.open();

  // Shared state for visualization
  brain best_brain_copy;
  std::mutex brain_mutex;
  std::atomic<bool> training_done{false};
  std::atomic<bool> new_generation{false};

  // Training coroutine - runs async, updates shared state
  auto training_task = [&]() -> ethreads::coro_task<void> {
    decltype(genome::fitness) current_best;
    do {
      co_await xor_model.train_async();

      current_best = xor_model.get_fitness(xor_model.best);

      // Copy best brain for visualization (thread-safe)
      {
        std::lock_guard<std::mutex> lock(brain_mutex);
        best_brain_copy = xor_model.best;
      }
      new_generation.store(true);

      std::cerr
          << "Generation: " << xor_model.p->generation_number
          << " Population: " << xor_model.p->speciating_parameters.population
          << " Unique Species: " << xor_model.p->species.size() << " Fitness: "
          << current_best /
                 (exfloat)std::numeric_limits<decltype(genome::fitness)>::max()
          << " Neuron Count: " << xor_model.best.neurons.size() << '\r';
    } while (current_best /
                 (exfloat)std::numeric_limits<decltype(genome::fitness)>::max() <
             needed_fitness);

    std::vector<exfloat> output = {0};

    std::cerr << std::endl;
    for (const auto &[test_input, expected_output] : test_cases) {
      xor_model.best.evaluate(test_input, output);
      std::cerr << test_input[0] << " " << test_input[1] << " -> "
                << (int)(output[0] + 0.5f) << std::endl;
    }

    std::cerr << "Training complete. Close visualization window to exit." << std::endl;
    training_done.store(true);
    co_return;
  };

  // Start training in background
  auto task = training_task();
  task.start();

  // Main thread render loop - keeps OpenGL context on main thread
  while (visualizer.is_open()) {
    // Check if new generation is ready to render
    if (new_generation.exchange(false)) {
      std::lock_guard<std::mutex> lock(brain_mutex);
      visualizer.render(best_brain_copy);
    } else if (training_done.load()) {
      // Training done, keep rendering final state
      std::lock_guard<std::mutex> lock(brain_mutex);
      visualizer.render(best_brain_copy);
    }
    visualizer.update();
  }

  // Wait for training to complete if window closed early
  if (!training_done.load()) {
    task.get();
  }

  return 0;
}
