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
  model sine_model(fitness_function, model_name, 1, 1, 100, 1, false);

  // Configure for diverse activation functions
  auto& rates = sine_model.p->mutation_rates;
  rates.activation_mutation_chance = 0.2f;  // High chance for diverse activations
  rates.link_mutation_chance = 0.2f;
  rates.neuron_mutation_chance = 0.02f;

  auto& params = sine_model.p->speciating_parameters;
  params.population = 100;
  params.target_species_count = 8;

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
    constexpr float target_fitness = 0.9f;

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

  // Get initial brain
  {
    auto best = sine_model.p->get_best_brain();
    if (best) {
      best_brain_copy.store(*best);
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

    // Check for brain update
    if (update_ready.try_wait()) {
      // Brain was updated, will be rendered in next frame
    }

    // Get current brain copy for rendering
    brain render_brain = best_brain_copy.load();

    // Apply override if active
    if (override_active.load() && !render_brain.output_neurons.empty()) {
      std::vector<exfloat> override_vals = {static_cast<exfloat>(override_value.load())};
      render_brain.override_outputs(override_vals);
    } else {
      render_brain.clear_overrides();
    }

    // Run a sample evaluation to update activations for visualization
    std::vector<exfloat> test_output(1, 0.0f);
    float test_x = std::fmod(static_cast<float>(current_tick.load()) * 0.01f, 1.0f);
    std::vector<exfloat> test_input = {static_cast<exfloat>(test_x)};
    render_brain.evaluate(test_input, test_output);

    // Begin rendering
    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Render network
    visualizer.render(render_brain);

    // Draw status information
    int y_offset = 10;

    // Tick and fitness
    char status[256];
    snprintf(status, sizeof(status), "Tick: %zu | Fitness: %.2f%%",
             current_tick.load(), current_fitness.load() * 100.0f);
    DrawText(status, 10, y_offset, 20, DARKGRAY);
    y_offset += 25;

    // Population info
    snprintf(status, sizeof(status), "Population: %zu | Species: %zu",
             sine_model.population_size(), sine_model.p->species.size());
    DrawText(status, 10, y_offset, 16, GRAY);
    y_offset += 20;

    // Network info
    snprintf(status, sizeof(status), "Neurons: %zu | outputs_ready: %s",
             render_brain.neurons.size(),
             render_brain.outputs_ready() ? "YES" : "NO");
    DrawText(status, 10, y_offset, 16, GRAY);
    y_offset += 25;

    // Override status
    if (override_active.load()) {
      snprintf(status, sizeof(status), "OVERRIDE ACTIVE: %.2f", override_value.load());
      DrawText(status, 10, y_offset, 18, RED);
    } else {
      DrawText("Override: OFF (press O to toggle)", 10, y_offset, 16, LIGHTGRAY);
    }
    y_offset += 25;

    // Pause status
    if (paused.load()) {
      DrawText("PAUSED (press SPACE to resume)", 10, y_offset, 18, ORANGE);
    }

    // Current test point visualization
    int graph_x = 10;
    int graph_y = GetScreenHeight() - 150;
    int graph_w = 300;
    int graph_h = 120;

    DrawRectangleLines(graph_x, graph_y, graph_w, graph_h, LIGHTGRAY);
    DrawText("Sine Approximation", graph_x, graph_y - 20, 14, DARKGRAY);

    // Draw actual sine wave
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

    // Draw network's approximation
    for (int i = 0; i < graph_w - 1; i++) {
      float x1 = static_cast<float>(i) / graph_w;
      float x2 = static_cast<float>(i + 1) / graph_w;

      std::vector<exfloat> out1(1), out2(1);
      render_brain.evaluate({static_cast<exfloat>(x1)}, out1);
      render_brain.evaluate({static_cast<exfloat>(x2)}, out2);

      DrawLine(
        graph_x + i,
        graph_y + graph_h - static_cast<int>(static_cast<float>(out1[0]) * graph_h),
        graph_x + i + 1,
        graph_y + graph_h - static_cast<int>(static_cast<float>(out2[0]) * graph_h),
        RED
      );
    }

    // Draw current test point
    int test_px = graph_x + static_cast<int>(test_x * graph_w);
    int test_py = graph_y + graph_h - static_cast<int>(static_cast<float>(test_output[0]) * graph_h);
    DrawCircle(test_px, test_py, 5, GREEN);

    // Legend
    DrawText("Blue: sin(x)  Red: Network  Green: Current", graph_x, graph_y + graph_h + 5, 12, GRAY);

    // Draw activation function legend
    int legend_x = GetScreenWidth() - 150;
    int legend_y = 10;
    DrawText("Activation Types:", legend_x, legend_y, 14, DARKGRAY);
    legend_y += 20;

    const char* act_names[] = {"ReLU", "Linear", "Step", "Logistic", "Sigmoid",
                               "Tanh", "GELU", "Swish", "LReLU", "Norm"};
    for (int i = 0; i < 10; i++) {
      DrawText(act_names[i], legend_x, legend_y, 12, GRAY);
      legend_y += 15;
    }

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
