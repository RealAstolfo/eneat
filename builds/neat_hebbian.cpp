// neat_hebbian.cpp - Hebbian Learning & Temporal Memory Demo
//
// This example demonstrates:
// - Hebbian learning: apply_hebbian_learning(), trait mutations, 8-parameter traits
// - Time-delayed recurrence: is_time_delayed, last_activation2, get_active_out_td()
// - Network state management: flush(), flushback(), reset_state(), outputs_ready()
// - Unused activation functions: GELU, SWISH, LEAKY_RELU

#include "neat.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "coro_task.hpp"
#include "trait.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>

#define needed_fitness 0.85

// T-Maze Environment
// The agent starts at position 0 and must navigate to the correct goal.
// At the start, a signal indicates which direction to go (LEFT or RIGHT).
// The agent must remember this signal and act on it when reaching the junction.
//
// Layout:
//   [L]---[J]---[R]
//          |
//         [S]  (start)
//
// Inputs: [signal, position, at_junction]
// - signal: -1.0 for LEFT, +1.0 for RIGHT (only visible at start)
// - position: normalized position along corridor (-1 to +1)
// - at_junction: 1.0 if at junction, 0.0 otherwise
//
// Output: action (-1 to +1, interpreted as direction)

struct TMaze {
  enum Goal { LEFT = -1, RIGHT = +1 };

  float agent_position;  // -1.0 (left goal) to +1.0 (right goal), 0.0 is junction
  float corridor_pos;    // 0.0 (start) to 1.0 (junction)
  Goal correct_goal;
  bool at_junction;
  bool signal_visible;
  int steps;
  static constexpr int max_steps = 50;

  TMaze() { reset(LEFT); }

  void reset(Goal goal) {
    correct_goal = goal;
    agent_position = 0.0f;
    corridor_pos = 0.0f;
    at_junction = false;
    signal_visible = true;  // Signal is only visible at the start
    steps = 0;
  }

  std::vector<exfloat> get_observation() const {
    float signal = signal_visible ? static_cast<float>(correct_goal) : 0.0f;
    return {
      static_cast<exfloat>(signal),
      static_cast<exfloat>(corridor_pos),
      static_cast<exfloat>(at_junction ? 1.0f : 0.0f)
    };
  }

  // Returns reward: +1.0 for correct goal, -1.0 for wrong goal, small penalty otherwise
  float step(float action) {
    steps++;

    // Signal disappears after first step
    if (steps > 1) {
      signal_visible = false;
    }

    // Move towards junction first
    if (!at_junction) {
      corridor_pos += 0.1f;
      if (corridor_pos >= 1.0f) {
        corridor_pos = 1.0f;
        at_junction = true;
      }
      return -0.01f;  // Small time penalty
    }

    // At junction: move based on action
    float move = action * 0.1f;
    agent_position += move;
    agent_position = std::clamp(agent_position, -1.0f, 1.0f);

    // Check if reached a goal
    if (agent_position <= -0.9f) {
      // Reached LEFT goal
      return (correct_goal == LEFT) ? 1.0f : -1.0f;
    } else if (agent_position >= 0.9f) {
      // Reached RIGHT goal
      return (correct_goal == RIGHT) ? 1.0f : -1.0f;
    }

    return -0.01f;  // Time penalty
  }

  bool is_done() const {
    return steps >= max_steps || std::abs(agent_position) >= 0.9f;
  }

  bool succeeded() const {
    if (agent_position <= -0.9f && correct_goal == LEFT) return true;
    if (agent_position >= 0.9f && correct_goal == RIGHT) return true;
    return false;
  }
};

int main() {
  std::cerr << "NEAT Hebbian Learning Demo - T-Maze Navigation" << std::endl;
  std::cerr << "================================================" << std::endl;
  std::cerr << "Features: Hebbian learning, time-delayed recurrence, network state management" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Task: Agent must remember direction signal and navigate to correct goal" << std::endl;
  std::cerr << std::endl;

  // Fitness function with Hebbian learning
  const auto fitness_function = [](brain &net) -> ethreads::coro_task<size_t> {
    // Enable Hebbian learning for this network
    net.hebbian_enabled = true;

    TMaze maze;
    float total_reward = 0.0f;
    int successes = 0;

    // Run multiple episodes to test memory and learning
    constexpr int num_episodes = 10;

    for (int episode = 0; episode < num_episodes; episode++) {
      // Alternate between LEFT and RIGHT goals
      TMaze::Goal goal = (episode % 2 == 0) ? TMaze::LEFT : TMaze::RIGHT;
      maze.reset(goal);

      // Flush network state between episodes (critical for temporal tasks)
      net.flush();

      // Wait for outputs to be ready (for proper network initialization)
      std::vector<exfloat> output(1, 0.0f);
      for (int warmup = 0; warmup < 3; warmup++) {
        auto obs = maze.get_observation();
        net.evaluate(obs, output);
      }

      // Run episode
      while (!maze.is_done()) {
        auto obs = maze.get_observation();
        net.evaluate(obs, output);

        // Convert output to action
        float action = std::clamp(static_cast<float>(output[0]), -1.0f, 1.0f);
        float reward = maze.step(action);
        total_reward += reward;
      }

      if (maze.succeeded()) {
        successes++;
      }
    }

    // Fitness based on success rate and total reward
    float success_rate = static_cast<float>(successes) / num_episodes;
    float normalized_reward = (total_reward + num_episodes) / (2.0f * num_episodes);

    // Combined fitness: 70% success rate, 30% reward
    float fitness = 0.7f * success_rate + 0.3f * std::max(0.0f, normalized_reward);

    co_return std::lerp(0.0f, (exfloat)std::numeric_limits<size_t>::max(), fitness);
  };

  std::string model_name = "tmaze";
  // 3 inputs, 1 output, 150 population, 1 bias, RECURRENT (essential for memory)
  model tmaze_model(fitness_function, model_name, 3, 1, 150, 1, true);

  // === Configure Hebbian learning and trait mutations ===
  auto& rates = tmaze_model.p->mutation_rates;

  // Enable trait mutations for evolving Hebbian learning parameters
  rates.trait_mutation_chance = 0.2f;           // Mutate trait parameters
  rates.link_trait_mutation_chance = 0.15f;     // Assign traits to links
  rates.node_trait_mutation_chance = 0.15f;     // Assign traits to nodes
  rates.trait_param_mutation_power = 0.5f;      // Trait parameter perturbation magnitude

  // Enable diverse activation functions
  rates.activation_mutation_chance = 0.15f;

  // Structural mutations (moderate for this task)
  rates.link_mutation_chance = 0.2f;
  rates.neuron_mutation_chance = 0.01f;

  auto& params = tmaze_model.p->speciating_parameters;
  params.population = 150;
  params.target_species_count = 12;
  params.dynamic_threshold_enabled = true;
  params.time_alive_minimum = 5;

  // === Create initial Hebbian learning traits ===
  std::cerr << "Creating initial Hebbian learning traits..." << std::endl;

  // Add traits to initial genomes
  for (auto& species : tmaze_model.p->species) {
    species.genomes.modify([](std::vector<genome>& genomes) {
      for (auto& g : genomes) {
        // Trait 1: No learning (control)
        eneat::trait no_learn(1, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        g.add_trait(no_learn);

        // Trait 2: Hebbian learning (enabled, trait_id 2 enables learning)
        // params: [hebb_rate, pre_rate, post_rate, max_weight, min_weight, learning_signal, r1, r2]
        eneat::trait hebbian(2, 0.1f, 0.05f, 0.05f, 5.0f, 0.0f, 0.5f, 0.0f, 0.0f);
        g.add_trait(hebbian);

        // Trait 3: Anti-Hebbian (negative correlation learning)
        eneat::trait anti_hebbian(3, -0.1f, 0.05f, -0.05f, 5.0f, 0.0f, 0.5f, 0.0f, 0.0f);
        g.add_trait(anti_hebbian);

        // Trait 4: Presynaptic-only learning
        eneat::trait presynaptic(4, 0.0f, 0.2f, 0.0f, 5.0f, 0.0f, 0.5f, 0.0f, 0.0f);
        g.add_trait(presynaptic);
      }
    });
  }

  std::cerr << "Trait mutation chance: " << rates.trait_mutation_chance << std::endl;
  std::cerr << "Link trait mutation: " << rates.link_trait_mutation_chance << std::endl;
  std::cerr << "Node trait mutation: " << rates.node_trait_mutation_chance << std::endl;
  std::cerr << std::endl;

  std::cerr << "Starting evolution with Hebbian learning..." << std::endl;
  std::cerr << std::endl;

  size_t tick = 0;
  size_t last_report_tick = 0;

  // Track Hebbian learning statistics
  size_t max_traits_seen = 0;
  size_t learning_networks = 0;

  while (tmaze_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max() < needed_fitness) {
    auto tick_task = tmaze_model.tick_async();
    tick_task.start();
    tick_task.get();

    // === Analyze Hebbian learning usage ===
    if (tick % 200 == 0) {
      auto best = tmaze_model.p->get_best_genome();
      if (best) {
        max_traits_seen = std::max(max_traits_seen, best->traits.size());

        // Count learning-enabled traits
        size_t learning_traits = 0;
        for (const auto& t : best->traits) {
          if (t.is_learning_enabled()) {
            learning_traits++;
          }
        }

        // Check if any genes have learning traits assigned
        bool has_learning = false;
        for (const auto& [_, gene] : best->genes) {
          if (gene.trait_id >= 2 && gene.trait_id <= 4) {
            has_learning = true;
            break;
          }
        }

        if (has_learning) learning_networks++;
      }
    }

    // === Progress report ===
    if (tick - last_report_tick >= 500) {
      last_report_tick = tick;
      float fitness = tmaze_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max();

      // Calculate success rate from fitness
      float approx_success = (fitness - 0.3f * 0.5f) / 0.7f;  // Reverse the fitness formula
      approx_success = std::clamp(approx_success, 0.0f, 1.0f);

      auto best = tmaze_model.p->get_best_genome();
      size_t traits = 0;
      size_t neurons = 0;
      if (best) {
        traits = best->traits.size();
        brain b;
        b = *best;
        neurons = b.neurons.size();
      }

      std::cerr << "[Tick " << tick << "] "
                << "Success: ~" << (approx_success * 100.0f) << "% | "
                << "Fitness: " << fitness << " | "
                << "Traits: " << traits << " | "
                << "Neurons: " << neurons << " | "
                << "Species: " << tmaze_model.p->species.size()
                << std::endl;
    }

    tick++;

    if (tick > 50000) {
      std::cerr << "Reached tick limit" << std::endl;
      break;
    }
  }

  // === Display final results ===
  std::cerr << std::endl;
  std::cerr << "Evolution complete in " << tick << " ticks" << std::endl;

  float final_fitness = tmaze_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max();
  std::cerr << "Final fitness: " << final_fitness << std::endl;
  std::cerr << "Max traits evolved: " << max_traits_seen << std::endl;
  std::cerr << std::endl;

  // === Test the best network ===
  std::cerr << "=== Testing Best Network ===" << std::endl;

  auto best_brain = tmaze_model.p->get_best_brain();
  if (best_brain) {
    std::cerr << "Network neurons: " << best_brain->neurons.size() << std::endl;
    std::cerr << "Hebbian enabled: " << (best_brain->hebbian_enabled ? "yes" : "no") << std::endl;
    std::cerr << "Traits: " << best_brain->traits.size() << std::endl;

    // Display activation function distribution
    std::map<ai_func_type, int> activation_counts;
    for (const auto& n : best_brain->neurons) {
      activation_counts[n.activation_function]++;
    }
    std::cerr << "Activation functions: ";
    for (const auto& [func, count] : activation_counts) {
      std::cerr << activation_name(func) << "=" << count << " ";
    }
    std::cerr << std::endl;
    std::cerr << std::endl;

    // Run test episodes
    TMaze maze;
    int total_successes = 0;
    constexpr int test_episodes = 20;

    best_brain->hebbian_enabled = true;

    std::cerr << "Running " << test_episodes << " test episodes..." << std::endl;

    for (int ep = 0; ep < test_episodes; ep++) {
      TMaze::Goal goal = (ep % 2 == 0) ? TMaze::LEFT : TMaze::RIGHT;
      maze.reset(goal);

      // Demonstrate flush() between episodes
      best_brain->flush();

      // Demonstrate outputs_ready() check
      std::vector<exfloat> output(1, 0.0f);

      // Warmup to ensure outputs are ready
      int warmup_steps = 0;
      while (!best_brain->outputs_ready() && warmup_steps < 5) {
        auto obs = maze.get_observation();
        best_brain->evaluate(obs, output);
        warmup_steps++;
      }

      // Run episode
      while (!maze.is_done()) {
        auto obs = maze.get_observation();
        best_brain->evaluate(obs, output);
        float action = std::clamp(static_cast<float>(output[0]), -1.0f, 1.0f);
        maze.step(action);
      }

      if (maze.succeeded()) {
        total_successes++;
      }

      // Show some episodes in detail
      if (ep < 4) {
        std::cerr << "  Episode " << ep + 1 << ": Goal="
                  << (goal == TMaze::LEFT ? "LEFT" : "RIGHT")
                  << ", Result=" << (maze.succeeded() ? "SUCCESS" : "FAIL")
                  << ", Position=" << maze.agent_position
                  << std::endl;
      }
    }

    std::cerr << std::endl;
    std::cerr << "Test success rate: " << total_successes << "/" << test_episodes
              << " (" << (total_successes * 100.0f / test_episodes) << "%)" << std::endl;

    // === Demonstrate reset_state() vs flush() ===
    std::cerr << std::endl;
    std::cerr << "=== Network State Management Demo ===" << std::endl;

    // Show initial state
    std::cerr << "After flush(): outputs_ready = " << (best_brain->outputs_ready() ? "yes" : "no") << std::endl;

    // Run a few steps
    std::vector<exfloat> test_output(1, 0.0f);
    for (int i = 0; i < 3; i++) {
      std::vector<exfloat> test_input = {0.0f, 0.0f, 0.0f};
      best_brain->evaluate(test_input, test_output);
    }
    std::cerr << "After 3 evaluations: outputs_ready = " << (best_brain->outputs_ready() ? "yes" : "no") << std::endl;

    // Demonstrate reset_state (full reset including overrides)
    best_brain->reset_state();
    std::cerr << "After reset_state(): outputs_ready = " << (best_brain->outputs_ready() ? "yes" : "no") << std::endl;

    // Demonstrate flushback on a specific neuron
    if (best_brain->output_neurons.size() > 0) {
      best_brain->flushback(best_brain->output_neurons[0]);
      std::cerr << "After flushback(output[0]): output activation_count = "
                << best_brain->neurons[best_brain->output_neurons[0]].activation_count << std::endl;
    }

    // === Display trait information ===
    std::cerr << std::endl;
    std::cerr << "=== Evolved Traits ===" << std::endl;
    for (const auto& t : best_brain->traits) {
      std::cerr << "Trait " << t.trait_id << ": ";
      std::cerr << "hebb=" << t.hebbian_rate() << " ";
      std::cerr << "pre=" << t.presynaptic_rate() << " ";
      std::cerr << "post=" << t.postsynaptic_rate() << " ";
      std::cerr << "max_w=" << t.max_weight() << " ";
      std::cerr << "learning=" << (t.is_learning_enabled() ? "yes" : "no");
      std::cerr << std::endl;
    }
  } else {
    std::cerr << "No best brain available" << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "Done!" << std::endl;

  return 0;
}
