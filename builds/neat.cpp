#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "coro_task.hpp"
#include <algorithm>
#include <cstddef>
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

  std::string model_name = "xor";
  // model(fitness_func, name, inputs, outputs, population, bias, recurrent)
  model xor_model(fitness_function, model_name, 2, 1, 1000, 1, false);
  xor_model.p->speciating_parameters.time_alive_minimum = 5;  // rtNEAT maturity
  
  // rtNEAT continuous evolution loop
  size_t current_best;
  size_t last_report_tick = 0;

  auto evolve_task = [&]() -> ethreads::coro_task<void> {
    do {
      // Run 10 ticks of evolution
      co_await xor_model.evolve_async(100);

      current_best = xor_model.get_best_fitness();
      size_t current_tick = xor_model.tick_count.load();

      // Report progress every 10 ticks
      if (current_tick - last_report_tick >= 10) {
        last_report_tick = current_tick;
        std::cerr
            << "Tick: " << current_tick
            << " Population: " << xor_model.population_size()
            << " Species: " << xor_model.p->species.size()
            << " Fitness: "
            << current_best / (exfloat)std::numeric_limits<size_t>::max()
            << " Neurons: " << xor_model.get_best_brain().neurons.size()
            << "        \r";
      }
    } while (current_best /
                 (exfloat)std::numeric_limits<size_t>::max() <
             needed_fitness);
    co_return;
  };

  auto task = evolve_task();
  task.start();
  task.get();

  std::vector<exfloat> output = {0};
  brain final_brain = xor_model.get_best_brain();

  std::cerr << std::endl;
  std::cerr << "Completed in " << xor_model.tick_count.load() << " ticks" << std::endl;
  for (const auto &[test_input, expected_output] : test_cases) {
    final_brain.evaluate(test_input, output);
    std::cerr << test_input[0] << " " << test_input[1] << " -> "
              << (int)(output[0] + 0.5f) << std::endl;
  }

  return 0;
}
