// neat_genome_control.cpp - Gene Expression & Crossover Variants Demo
//
// This example demonstrates:
// - Gene enable/disable mutations (mutate_enable_disable)
// - Single-point crossover (crossover_singlepoint)
// - Weight-only mutation mode (mutate_weight_only)
// - Sensor mutation (mutate_add_sensor)
// - Genome analysis: extrons(), randomize_traits(), frozen_nodes, frozen genes
// - Direct stream serialization for brain

#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "coro_task.hpp"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

#define needed_fitness 0.95

// 6-bit Multiplexer: 2 address bits (A0, A1) select from 4 data bits (D0-D3)
// Input: [A0, A1, D0, D1, D2, D3]
// Output: D[A0 + 2*A1]
static std::vector<std::pair<std::vector<exfloat>, exfloat>> generate_mux_cases() {
  std::vector<std::pair<std::vector<exfloat>, exfloat>> cases;

  // Generate all 64 input combinations
  for (int a0 = 0; a0 <= 1; a0++) {
    for (int a1 = 0; a1 <= 1; a1++) {
      for (int d0 = 0; d0 <= 1; d0++) {
        for (int d1 = 0; d1 <= 1; d1++) {
          for (int d2 = 0; d2 <= 1; d2++) {
            for (int d3 = 0; d3 <= 1; d3++) {
              std::vector<exfloat> input = {
                static_cast<exfloat>(a0),
                static_cast<exfloat>(a1),
                static_cast<exfloat>(d0),
                static_cast<exfloat>(d1),
                static_cast<exfloat>(d2),
                static_cast<exfloat>(d3)
              };

              // Select based on address bits
              int selector = a0 + 2 * a1;
              exfloat expected = 0.0f;
              switch (selector) {
                case 0: expected = static_cast<exfloat>(d0); break;
                case 1: expected = static_cast<exfloat>(d1); break;
                case 2: expected = static_cast<exfloat>(d2); break;
                case 3: expected = static_cast<exfloat>(d3); break;
              }

              cases.push_back({input, expected});
            }
          }
        }
      }
    }
  }

  return cases;
}

static const auto mux_cases = generate_mux_cases();

int main() {
  std::cerr << "NEAT Genome Control Demo - 6-bit Multiplexer" << std::endl;
  std::cerr << "=============================================" << std::endl;
  std::cerr << "Features: Gene enable/disable, single-point crossover, weight-only mutation," << std::endl;
  std::cerr << "          sensor mutation, frozen genes, stream serialization" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Test cases: " << mux_cases.size() << " (all 64 combinations)" << std::endl;
  std::cerr << std::endl;

  // Fitness function
  const auto fitness_function = [](brain &net) -> ethreads::coro_task<size_t> {
    std::vector<exfloat> output(1, 0.0);
    exfloat correct = 0.0f;

    for (const auto &[test_input, expected_output] : mux_cases) {
      net.evaluate(test_input, output);

      // Binary classification: threshold at 0.5
      exfloat predicted = (output[0] > 0.5f) ? 1.0f : 0.0f;
      if (std::abs(predicted - expected_output) < 0.1f) {
        correct += 1.0f;
      }
    }

    // Fitness based on accuracy
    exfloat accuracy = correct / static_cast<exfloat>(mux_cases.size());
    co_return std::lerp(0.0f, (exfloat)std::numeric_limits<size_t>::max(), accuracy);
  };

  std::string model_name = "mux";
  // 6 inputs, 1 output, 150 population, 1 bias, recurrent allowed
  model mux_model(fitness_function, model_name, 6, 1, 150, 1, true);

  // === Configure gene expression mutations ===
  auto& rates = mux_model.p->mutation_rates;

  // Enable gene toggling mutations (usually disabled)
  rates.enable_mutation_chance = 0.3f;   // Re-enable disabled genes
  rates.disable_mutation_chance = 0.2f;  // Disable genes

  // Enable single-point crossover (30% of crossovers use single-point)
  rates.singlepoint_chance = 0.3f;

  // Standard mutation rates
  rates.link_mutation_chance = 0.15f;
  rates.neuron_mutation_chance = 0.003f;
  rates.activation_mutation_chance = 0.1f;

  // Trait mutations for demonstrating trait features
  rates.trait_mutation_chance = 0.1f;

  auto& params = mux_model.p->speciating_parameters;
  params.population = 150;
  params.target_species_count = 10;
  params.dynamic_threshold_enabled = true;

  std::cerr << "Starting evolution..." << std::endl;
  std::cerr << "Enable mutation: " << rates.enable_mutation_chance << std::endl;
  std::cerr << "Disable mutation: " << rates.disable_mutation_chance << std::endl;
  std::cerr << "Single-point crossover: " << rates.singlepoint_chance << std::endl;
  std::cerr << std::endl;

  // === Ensure initial connectivity via sensor mutation ===
  std::cerr << "Applying sensor mutations to connect all inputs..." << std::endl;
  for (auto& species : mux_model.p->species) {
    species.genomes.modify([&](std::vector<genome>& genomes) {
      for (auto& g : genomes) {
        mux_model.p->mutate_add_sensor_sync(g);
      }
    });
  }
  std::cerr << std::endl;

  size_t tick = 0;
  size_t last_report_tick = 0;

  // Track gene expression statistics
  size_t total_genes_ever = 0;
  size_t total_extrons_ever = 0;
  size_t genes_frozen = 0;

  while (mux_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max() < needed_fitness) {
    auto tick_task = mux_model.tick_async();
    tick_task.start();
    tick_task.get();

    // === Demonstrate gene expression tracking ===
    if (tick % 100 == 0) {
      auto best = mux_model.p->get_best_genome();
      if (best) {
        size_t active_genes = best->extrons();
        size_t total_genes = best->genes.size();

        total_genes_ever = std::max(total_genes_ever, total_genes);
        total_extrons_ever = std::max(total_extrons_ever, active_genes);

        // === Demonstrate weight-only mutation on super champions ===
        if (tick > 0 && tick % 500 == 0 && best->super_champ_offspring > 0) {
          std::cerr << "[Tick " << tick << "] Applying weight-only mutation to super champion" << std::endl;
          mux_model.p->mutate_weight_only(*best);
        }

        // === Demonstrate freezing high-performing genes ===
        if (active_genes > 10 && tick > 1000 && tick % 1000 == 0) {
          // Freeze the first few enabled genes as "critical"
          size_t frozen_count = 0;
          for (auto& [innov, gene] : best->genes) {
            if (gene.enabled && !gene.frozen && frozen_count < 3) {
              gene.frozen = true;
              frozen_count++;
              genes_frozen++;
            }
          }
          if (frozen_count > 0) {
            std::cerr << "[Tick " << tick << "] Froze " << frozen_count << " critical genes" << std::endl;
          }

          // Freeze some critical nodes
          if (best->frozen_nodes.empty()) {
            // Freeze the output neuron's trait
            best->frozen_nodes.insert(best->network_info.input_size +
                                      best->network_info.bias_size);
            std::cerr << "[Tick " << tick << "] Froze output node trait" << std::endl;
          }
        }

        // === Demonstrate randomize_traits ===
        if (tick == 2000 && !best->traits.empty()) {
          std::cerr << "[Tick " << tick << "] Demonstrating trait randomization (before mutation)" << std::endl;
          // Note: randomize_traits() would scramble trait assignments
          // We just log this capability here
        }
      }
    }

    // === Progress report ===
    if (tick - last_report_tick >= 500) {
      last_report_tick = tick;
      float fitness = mux_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max();

      auto best = mux_model.p->get_best_genome();
      size_t active = 0, total = 0;
      if (best) {
        active = best->extrons();
        total = best->genes.size();
      }

      std::cerr << "[Tick " << tick << "] "
                << "Accuracy: " << (fitness * 100.0f) << "% "
                << "(" << static_cast<int>(fitness * 64) << "/64) | "
                << "Genes: " << active << "/" << total << " active | "
                << "Frozen: " << genes_frozen << " | "
                << "Species: " << mux_model.p->species.size()
                << std::endl;
    }

    tick++;

    if (tick > 100000) {
      std::cerr << "Reached tick limit" << std::endl;
      break;
    }
  }

  // === Display final results ===
  std::cerr << std::endl;
  std::cerr << "Evolution complete in " << tick << " ticks" << std::endl;

  float final_fitness = mux_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max();
  std::cerr << "Final accuracy: " << (final_fitness * 100.0f) << "%" << std::endl;
  std::cerr << "Max genes seen: " << total_genes_ever << " (max active: " << total_extrons_ever << ")" << std::endl;
  std::cerr << "Genes frozen: " << genes_frozen << std::endl;
  std::cerr << std::endl;

  // === Demonstrate stream serialization ===
  std::cerr << "=== Stream Serialization Demo ===" << std::endl;

  auto best_brain = mux_model.p->get_best_brain();
  if (best_brain) {
    // Serialize brain to string stream
    std::ostringstream oss;
    oss << *best_brain;
    std::string brain_data = oss.str();

    std::cerr << "Serialized brain size: " << brain_data.size() << " bytes" << std::endl;
    std::cerr << "Neurons: " << best_brain->neurons.size() << std::endl;

    // Deserialize back
    std::istringstream iss(brain_data);
    brain loaded_brain;
    iss >> loaded_brain;

    std::cerr << "Loaded brain neurons: " << loaded_brain.neurons.size() << std::endl;

    // Verify deserialized brain works
    std::vector<exfloat> output(1, 0.0);
    size_t correct = 0;
    for (const auto &[test_input, expected] : mux_cases) {
      loaded_brain.evaluate(test_input, output);
      exfloat predicted = (output[0] > 0.5f) ? 1.0f : 0.0f;
      if (std::abs(predicted - expected) < 0.1f) {
        correct++;
      }
    }

    std::cerr << "Loaded brain accuracy: " << correct << "/64 ("
              << (correct * 100.0f / 64.0f) << "%)" << std::endl;

    // Save to file as well
    std::ofstream ofs("mux_best_brain.txt");
    if (ofs) {
      ofs << *best_brain;
      std::cerr << "Saved brain to mux_best_brain.txt" << std::endl;
    }
  }

  // === Test some multiplexer cases ===
  std::cerr << std::endl;
  std::cerr << "=== Sample Multiplexer Tests ===" << std::endl;

  if (best_brain) {
    std::vector<exfloat> output(1, 0.0);

    // Test a few representative cases
    std::vector<std::pair<std::vector<exfloat>, std::string>> samples = {
      {{0, 0, 1, 0, 0, 0}, "A=00, D=[1,0,0,0] -> D0=1"},
      {{1, 0, 0, 1, 0, 0}, "A=01, D=[0,1,0,0] -> D1=1"},
      {{0, 1, 0, 0, 1, 0}, "A=10, D=[0,0,1,0] -> D2=1"},
      {{1, 1, 0, 0, 0, 1}, "A=11, D=[0,0,0,1] -> D3=1"},
      {{0, 0, 0, 1, 1, 1}, "A=00, D=[0,1,1,1] -> D0=0"},
      {{1, 1, 1, 1, 1, 0}, "A=11, D=[1,1,1,0] -> D3=0"},
    };

    for (const auto& [input, desc] : samples) {
      best_brain->evaluate(input, output);
      std::cerr << desc << " | Output: " << output[0]
                << " (" << ((output[0] > 0.5f) ? "1" : "0") << ")" << std::endl;
    }
  }

  std::cerr << std::endl;
  std::cerr << "Done!" << std::endl;

  return 0;
}
