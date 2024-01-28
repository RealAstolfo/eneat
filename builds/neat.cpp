#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <time.h>

#define xor_max_n 4
#define needed_fitness 0.95

int main() {
  const auto fitness_function = [](brain &net) -> decltype(genome::fitness) {
    if (net.neurons.size() > xor_max_n)
      return 0.0f;

    std::vector<exfloat> output(1, 0.0);
    exfloat fitness = 0.0f;
    const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
        {{0.0, 0.0}, 0.0f},
        {{0.0, 1.0}, 1.0f},
        {{1.0, 0.0}, 1.0f},
        {{1.0, 1.0}, 0.0f}};

    for (const auto &[test_input, expected_output] : test_cases) {
      net.evaluate(test_input, output);
      fitness += std::lerp(1.0f, 0.0f,
                           std::min(1.0f, error(output[0], expected_output)));
    }

    return std::lerp(0.0f,
                     std::numeric_limits<decltype(genome::fitness)>::max(),
                     fitness / 4.0f);
  };

  std::string model_name = "xor";
  model xor_model(fitness_function, model_name);
  xor_model.p->speciating_parameters.population = 50;
  xor_model.p->speciating_parameters.stale_species = std::numeric_limits<
      decltype(xor_model.p->speciating_parameters.stale_species)>::max();

  decltype(genome::fitness) current_best;
  do {
    xor_model.train();
    current_best = xor_model.get_fitness(xor_model.best);
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
  const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
      {{0.0, 0.0}, 0.0f},
      {{0.0, 1.0}, 1.0f},
      {{1.0, 0.0}, 1.0f},
      {{1.0, 1.0}, 0.0f}};

  std::cerr << std::endl;
  for (const auto &[test_input, expected_output] : test_cases) {
    xor_model.best.evaluate(test_input, output);
    std::cerr << test_input[0] << " " << test_input[1] << " -> "
              << (int)(output[0] + 0.5f) << std::endl;
  }

  return 0;
}
