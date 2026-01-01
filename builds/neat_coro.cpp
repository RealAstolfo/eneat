#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "network_visualizer.hpp"
#include "shared_state.hpp"
#include <limits>

#define xor_max_n 4
#define needed_fitness 0.95

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
  model xor_model(fitness_function, model_name);
  xor_model.p->speciating_parameters.population = 50;
  xor_model.p->speciating_parameters.stale_species = std::numeric_limits<
      decltype(xor_model.p->speciating_parameters.stale_species)>::max();

  // Initialize visualizer on main thread
  NetworkVisualizer visualizer;
  visualizer.open();

  // Shared state for visualization
  ethreads::sync_shared_value<brain> best_brain_copy{};
  ethreads::manual_reset_event training_done{false};
  ethreads::auto_reset_event new_generation{false};

  // Training coroutine - runs async, updates shared state
  auto training_task = [&]() -> ethreads::coro_task<void> {
    size_t current_best;
    do {
      co_await xor_model.train_async();

      current_best = xor_model.get_best_fitness();

      // Copy best brain for visualization (thread-safe)
      best_brain_copy.store(xor_model.get_best_brain());
      new_generation.set();

      std::cerr
          << "Generation: " << xor_model.p->generation_number.load()
          << " Population: " << xor_model.p->speciating_parameters.population
          << " Unique Species: " << xor_model.p->species.size() << " Fitness: "
          << current_best /
                 (exfloat)std::numeric_limits<size_t>::max()
          << " Neuron Count: " << xor_model.get_best_brain().neurons.size() << '\r';
    } while (current_best /
                 (exfloat)std::numeric_limits<size_t>::max() <
             needed_fitness);

    std::vector<exfloat> output = {0};
    brain final_brain = xor_model.get_best_brain();

    std::cerr << std::endl;
    for (const auto &[test_input, expected_output] : test_cases) {
      final_brain.evaluate(test_input, output);
      std::cerr << test_input[0] << " " << test_input[1] << " -> "
                << (int)(output[0] + 0.5f) << std::endl;
    }

    std::cerr << "Training complete. Close visualization window to exit." << std::endl;
    training_done.set();
    co_return;
  };

  // Start training in background
  auto task = training_task();
  task.start();

  // Main thread render loop - keeps OpenGL context on main thread
  while (visualizer.is_open()) {
    // Check if new generation is ready to render
    if (new_generation.try_wait()) {
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
