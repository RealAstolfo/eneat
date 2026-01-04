// neat_visualized.cpp - Interactive Visualization & Output Override Demo
//
// This example demonstrates:
// - Visualization customization: set_label_callback() for custom neuron labels
// - Output override: override_outputs(), clear_overrides()
// - Display all activation functions: GELU, SWISH, LEAKY_RELU, HEAVISIDE, NORMALIZE
// - Network state: real-time outputs_ready() display

#include "neat.hpp"
#include "network_visualizer.hpp"
#include "math.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "coro_task.hpp"
#include "shared_state.hpp"
#include "shared_state/event.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>
#include <thread>
#include <atomic>
#include <sstream>
#include <iomanip>

// Sine wave approximation task
// Network learns to approximate sin(x) for x in [0, 2*pi]
static constexpr float MY_PI = 3.14159265358979323846f;

static std::vector<std::pair<std::vector<exfloat>, exfloat>> generate_sine_cases() {
  std::vector<std::pair<std::vector<exfloat>, exfloat>> cases;

  // Generate training points
  for (int i = 0; i < 32; i++) {
    float x = (static_cast<float>(i) / 31.0f) * 2.0f * MY_PI;
    float y = std::sin(x);

    // Normalize x to [0, 1] and y to [0, 1]
    float x_norm = x / (2.0f * MY_PI);
    float y_norm = (y + 1.0f) / 2.0f;

    cases.push_back({{static_cast<exfloat>(x_norm)}, static_cast<exfloat>(y_norm)});
  }

  return cases;
}

static const auto sine_cases = generate_sine_cases();

// Custom label callback that shows activation function and value
void draw_custom_label(size_t idx, float x, float y, const neuron& n) {
  // Get activation function name
  const char* func_name = activation_name(n.activation_function);

  // Format value
  char value_str[32];
  snprintf(value_str, sizeof(value_str), "%.2f", static_cast<double>(n.value));

  // Draw activation function name above neuron
  DrawText(func_name, static_cast<int>(x - 15), static_cast<int>(y - 25), 10, BLACK);

  // Draw current value below neuron
  DrawText(value_str, static_cast<int>(x - 15), static_cast<int>(y + 15), 10, DARKGRAY);

  // Draw neuron index
  char idx_str[16];
  snprintf(idx_str, sizeof(idx_str), "#%zu", idx);
  DrawText(idx_str, static_cast<int>(x + 15), static_cast<int>(y - 10), 8, GRAY);

  // Show trait ID if present
  if (n.trait_id > 0) {
    char trait_str[16];
    snprintf(trait_str, sizeof(trait_str), "T%zu", n.trait_id);
    DrawText(trait_str, static_cast<int>(x + 15), static_cast<int>(y + 5), 8, BLUE);
  }

  // Show override indicator
  if (n.override_active) {
    DrawCircle(static_cast<int>(x + 20), static_cast<int>(y + 15), 4, RED);
  }
}

int main() {
  std::cerr << "NEAT Interactive Visualization Demo" << std::endl;
  std::cerr << "====================================" << std::endl;
  std::cerr << "Features: Custom labels, output override, all activation functions" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Controls:" << std::endl;
  std::cerr << "  O - Toggle output override mode" << std::endl;
  std::cerr << "  Mouse Y - Adjust override value (when override is ON)" << std::endl;
  std::cerr << "  R - Reset network state" << std::endl;
  std::cerr << "  SPACE - Pause/resume training" << std::endl;
  std::cerr << "  ESC - Exit" << std::endl;
  std::cerr << std::endl;

  // Fitness function
  const auto fitness_function = [](brain &net) -> ethreads::coro_task<size_t> {
    std::vector<exfloat> output(1, 0.0);
    exfloat total_error = 0.0f;

    for (const auto &[test_input, expected_output] : sine_cases) {
      net.evaluate(test_input, output);
      total_error += std::abs(output[0] - expected_output);
    }

    // Fitness is inverse of error
    float avg_error = static_cast<float>(total_error) / sine_cases.size();
    float fitness = 1.0f - std::min(1.0f, avg_error);

    co_return std::lerp(0.0f, (exfloat)std::numeric_limits<size_t>::max(), fitness);
  };

  std::string model_name = "sine";
  // 1 input, 1 output, 100 population, 1 bias, non-recurrent
  model sine_model(fitness_function, model_name, 1, 1, 1000, 1, true);

  // Configure mutation rates for function approximation
  auto& rates = sine_model.p->mutation_rates;
  rates.connection_mutate_chance = 0.9f;    // High weight mutation for fine-tuning
  rates.step_size = 2.0f;                   // Weight adjustment power
  rates.activation_mutation_chance = 0.15f; // Diverse activations
  rates.link_mutation_chance = 0.3f;        // More connections for complexity
  rates.neuron_mutation_chance = 0.05f;     // Need hidden neurons for sine

  auto& params = sine_model.p->speciating_parameters;
  params.population = 1000;
  params.delta_coding_enabled = true;        // Enable delta coding for stagnation recovery
  params.dynamic_threshold_enabled = true;   // Dynamic compatibility threshold
  params.target_species_count = 15;          // Target species for diversity
  params.compat_adjust_frequency = 5;        // Adjust threshold more frequently
  params.dropoff_age = 50;                   // Faster species turnover

  // === Visualization setup ===
  NetworkVisualizer visualizer(1600, 1000);
  visualizer.open();

  // Set custom label callback
  visualizer.set_label_callback(draw_custom_label);

  // Shared state for thread communication
  ethreads::sync_shared_value<brain> best_brain_copy;
  ethreads::auto_reset_event update_ready;
  ethreads::manual_reset_event training_done;
  std::atomic<bool> paused{false};
  std::atomic<bool> should_exit{false};
  std::atomic<size_t> current_tick{0};
  std::atomic<float> current_fitness{0.0f};

  // Override control
  std::atomic<bool> override_active{false};
  std::atomic<float> override_value{0.5f};

  // Training task (runs in background)
  auto training_task = [&]() -> ethreads::coro_task<void> {
    constexpr float target_fitness = 0.99f;

    while (!should_exit.load() &&
           sine_model.p->max_fitness.load() / (exfloat)std::numeric_limits<size_t>::max() < target_fitness) {

      // Check for pause
      while (paused.load() && !should_exit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      if (should_exit.load()) break;

      // Run evolution tick
      co_await sine_model.tick_async();

      size_t tick = current_tick.fetch_add(1);

      // Update shared brain copy periodically
      if (tick % 10 == 0) {
        auto best = sine_model.p->get_best_brain();
        if (best) {
          best_brain_copy.store(*best);
          current_fitness.store(
            static_cast<float>(sine_model.p->max_fitness.load()) /
            static_cast<float>(std::numeric_limits<size_t>::max())
          );
          update_ready.set();
        }
      }
    }

    training_done.set();
    co_return;
  };

  // Start training in background
  auto task = training_task();
  task.start();

  std::cerr << "Training started. Visualizing..." << std::endl;
  std::cerr << std::endl;

  // Get initial brain and cache for stable rendering
  brain cached_render_brain;
  {
    auto best = sine_model.p->get_best_brain();
    if (best) {
      best_brain_copy.store(*best);
      cached_render_brain = *best;
    }
  }

  // === Main visualization loop ===
  while (visualizer.is_open() && !training_done.is_set()) {
    // Handle input
    if (IsKeyPressed(KEY_ESCAPE)) {
      should_exit.store(true);
      break;
    }

    if (IsKeyPressed(KEY_O)) {
      override_active.store(!override_active.load());
      std::cerr << "Override mode: " << (override_active.load() ? "ON" : "OFF") << std::endl;
    }

    if (IsKeyPressed(KEY_R)) {
      auto brain_copy = best_brain_copy.load();
      brain_copy.reset_state();
      best_brain_copy.store(brain_copy);
      std::cerr << "Network state reset" << std::endl;
    }

    if (IsKeyPressed(KEY_SPACE)) {
      paused.store(!paused.load());
      std::cerr << (paused.load() ? "Paused" : "Resumed") << std::endl;
    }

    // Mouse-based override value adjustment
    if (override_active.load() && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      float mouse_y = static_cast<float>(GetMouseY());
      float height = static_cast<float>(GetScreenHeight());
      override_value.store(1.0f - (mouse_y / height));  // Invert so up = higher value
    }

    // Check for brain update - only update cached brain when signaled
    if (update_ready.try_wait()) {
      cached_render_brain = best_brain_copy.load();
    }

    // Apply override if active
    if (override_active.load() && !cached_render_brain.output_neurons.empty()) {
      std::vector<exfloat> override_vals = {static_cast<exfloat>(override_value.load())};
      cached_render_brain.override_outputs(override_vals);
    } else {
      cached_render_brain.clear_overrides();
    }

    // Run a sample evaluation to update activations for visualization
    std::vector<exfloat> test_output(1, 0.0f);
    float test_x = std::fmod(static_cast<float>(current_tick.load()) * 0.01f, 1.0f);
    std::vector<exfloat> test_input = {static_cast<exfloat>(test_x)};
    cached_render_brain.evaluate(test_input, test_output);

    // Begin rendering
    BeginDrawing();
    ClearBackground(Color{30, 30, 30, 255});  // Dark background to match visualizer

    // Render network (without Begin/EndDrawing since we're managing the frame)
    visualizer.render_network_only(cached_render_brain);

    // Draw status information (positioned to not overlap with network visualizer legend)
    int status_x = 10;
    int status_y = GetScreenHeight() - 320;  // Above the graph
    char status[256];

    snprintf(status, sizeof(status), "Tick: %zu | Fitness: %.2f%%",
             current_tick.load(), current_fitness.load() * 100.0f);
    DrawText(status, status_x, status_y, 16, LIGHTGRAY);
    status_y += 20;

    snprintf(status, sizeof(status), "Population: %zu | Species: %zu | Neurons: %zu",
             sine_model.population_size(), sine_model.p->species.size(),
             cached_render_brain.neurons.size());
    DrawText(status, status_x, status_y, 14, GRAY);
    status_y += 18;

    if (override_active.load()) {
      snprintf(status, sizeof(status), "OVERRIDE: %.2f (O to toggle)", override_value.load());
      DrawText(status, status_x, status_y, 14, RED);
    } else {
      DrawText("O: Toggle override | R: Reset | SPACE: Pause", status_x, status_y, 12, DARKGRAY);
    }
    status_y += 18;

    if (paused.load()) {
      DrawText("PAUSED", status_x, status_y, 16, ORANGE);
    }

    // Sine approximation graph (bottom-left, with proper margins)
    int graph_x = 10;
    int graph_y = GetScreenHeight() - 140;
    int graph_w = 280;
    int graph_h = 100;

    // Background for graph
    DrawRectangle(graph_x - 2, graph_y - 22, graph_w + 4, graph_h + 40, Color{20, 20, 20, 200});
    DrawRectangleLines(graph_x, graph_y, graph_w, graph_h, LIGHTGRAY);
    DrawText("Sine Approximation", graph_x, graph_y - 18, 12, LIGHTGRAY);

    // Draw actual sine wave (blue)
    for (int i = 0; i < graph_w - 1; i++) {
      float x1 = static_cast<float>(i) / graph_w;
      float x2 = static_cast<float>(i + 1) / graph_w;
      float y1 = (std::sin(x1 * 2 * MY_PI) + 1.0f) / 2.0f;
      float y2 = (std::sin(x2 * 2 * MY_PI) + 1.0f) / 2.0f;

      DrawLine(
        graph_x + i,
        graph_y + graph_h - static_cast<int>(y1 * graph_h),
        graph_x + i + 1,
        graph_y + graph_h - static_cast<int>(y2 * graph_h),
        BLUE
      );
    }

    // Draw network's approximation (red) - use a fresh copy with clean state
    brain graph_brain = cached_render_brain;
    graph_brain.reset_state();     // Ensure clean state for evaluation
    graph_brain.clear_overrides(); // Remove any overrides

    // Pre-evaluate all points for smooth rendering
    std::vector<float> net_outputs(graph_w);
    std::vector<exfloat> out(1);
    for (int i = 0; i < graph_w; i++) {
      float x = static_cast<float>(i) / graph_w;
      graph_brain.evaluate({static_cast<exfloat>(x)}, out);
      net_outputs[i] = static_cast<float>(out[0]);
    }

    // Draw the approximation curve
    for (int i = 0; i < graph_w - 1; i++) {
      DrawLine(
        graph_x + i,
        graph_y + graph_h - static_cast<int>(net_outputs[i] * graph_h),
        graph_x + i + 1,
        graph_y + graph_h - static_cast<int>(net_outputs[i + 1] * graph_h),
        RED
      );
    }

    // Draw current test point
    int test_px = graph_x + static_cast<int>(test_x * graph_w);
    int test_py = graph_y + graph_h - static_cast<int>(static_cast<float>(test_output[0]) * graph_h);
    DrawCircle(test_px, test_py, 4, GREEN);

    // Legend
    DrawText("Blue=sin(x) Red=Network", graph_x, graph_y + graph_h + 4, 10, GRAY);

    EndDrawing();

    visualizer.update();
  }

  // Cleanup
  should_exit.store(true);
  task.get();  // Wait for training to complete

  visualizer.close();

  // === Display final results ===
  std::cerr << std::endl;
  std::cerr << "=== Final Results ===" << std::endl;
  std::cerr << "Ticks: " << current_tick.load() << std::endl;
  std::cerr << "Fitness: " << (current_fitness.load() * 100.0f) << "%" << std::endl;

  auto final_brain = sine_model.p->get_best_brain();
  if (final_brain) {
    std::cerr << "Neurons: " << final_brain->neurons.size() << std::endl;

    // Count activation functions
    std::map<ai_func_type, int> act_counts;
    for (const auto& n : final_brain->neurons) {
      act_counts[n.activation_function]++;
    }

    std::cerr << "Activation distribution:" << std::endl;
    for (const auto& [func, count] : act_counts) {
      std::cerr << "  " << activation_name(func) << ": " << count << std::endl;
    }

    // Test accuracy
    float total_error = 0.0f;
    std::vector<exfloat> output(1);
    for (const auto& [input, expected] : sine_cases) {
      final_brain->evaluate(input, output);
      total_error += std::abs(static_cast<float>(output[0]) - static_cast<float>(expected));
    }

    std::cerr << "Average error: " << (total_error / sine_cases.size()) << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "Done!" << std::endl;

  return 0;
}
