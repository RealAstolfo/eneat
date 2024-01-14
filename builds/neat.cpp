#include "neat.hpp"
#include "math.hpp"
#include <random>
#include <time.h>

size_t xor_test(brain &n) {
  std::vector<exfloat> input(2, 0.0);
  std::vector<exfloat> output(1, 0.0);
  size_t fitness = 0;
  exfloat answer;
  input[0] = 0.0, input[1] = 0.0, answer = 0.0;
  n.evaluate(input, output);
  fitness +=
      std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);
  input[0] = 0.0, input[1] = 1.0, answer = 1.0;
  n.evaluate(input, output);
  fitness +=
      std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);
  input[0] = 1.0, input[1] = 0.0, answer = 1.0;
  n.evaluate(input, output);
  fitness +=
      std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);
  input[0] = 1.0, input[1] = 1.0, answer = 0.0;
  n.evaluate(input, output);
  fitness +=
      std::min(1.0 / ((answer - output[0]) * (answer - output[0])), 50.0);
  return fitness;
}

/*
void test_output(){
        ann::neuralnet n;
        n.import_fromfile("fit = 200");
        xor_test(n, true);
}
*/

#define xor_max_n 4

int main() {
  pool p(2, 1, 0, false);
  srand(time(NULL));
  size_t max_fitness = 0;
  while (max_fitness < 200) {
    size_t brain_size = 0;
    size_t current_fitness = 0;
    size_t min_fitness = 100000;
    for (auto s = p.species.begin(); s != p.species.end(); s++) {
      for (size_t i = 0; i < (*s).genomes.size(); i++) {
        brain n;
        genome &g = (*s).genomes[i];
        n = g;
        brain_size = n.neurons.size();
        current_fitness = xor_test(n);
        if (n.neurons.size() > xor_max_n) { // penalize larger brains
          size_t cost = (n.neurons.size() - xor_max_n) * 10.0f;
          if (cost > current_fitness)
            current_fitness = 0;
          else
            current_fitness -= cost;
        }

        if (current_fitness < min_fitness)
          min_fitness = current_fitness;
        if (current_fitness > max_fitness)
          max_fitness = current_fitness;

        g.fitness = current_fitness;
        std::cerr << "Brain Size: " << brain_size
                  << " Fitness: " << current_fitness << std::endl;
      }
    }

    p.new_generation();
  }

  return 0;
}
