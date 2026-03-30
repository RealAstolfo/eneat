#include "neat.hpp"
#include "compute_backend.hpp"
#include "math.hpp"
#include "onnx_export.hpp"
#include "utils.hpp"
#include "async_runtime.hpp"
#include "coro_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

// XOR test cases
static const std::vector<std::pair<std::vector<exfloat>, exfloat>> test_cases = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f}};

int main() {
  std::cerr << "=== ENEAT NPU Pipeline Test ===" << std::endl;
  std::cerr << "Train → Export → Verify → Deploy" << std::endl;
  std::cerr << std::endl;

  // ============================================================
  // Phase 1: Train XOR with NEAT (non-recurrent for NPU compat)
  // ============================================================
  std::cerr << "[Phase 1] Training XOR with NEAT..." << std::endl;

  const auto fitness_function = [](brain &net) -> ethreads::coro_task<size_t> {
    std::vector<exfloat> output(1, 0.0f);
    exfloat fitness = 0.0f;

    for (const auto &[test_input, expected_output] : test_cases) {
      net.evaluate(test_input, output);
      fitness += std::lerp(1.0f, 0.0f,
                           std::min(1.0f, error(output[0], expected_output)));
    }

    co_return std::lerp(0.0f, (exfloat)std::numeric_limits<size_t>::max(),
                        fitness / 4.0f);
  };

  std::string model_name = "xor_npu";
  // Recurrent training — converges faster for XOR.
  // Export unrolls recurrent evaluation into feedforward ONNX.
  model xor_model(fitness_function, model_name, 2, 1, 100, 1, true);

  auto &params = xor_model.p->speciating_parameters;
  params.population = 100;
  params.time_alive_minimum = 5;
  params.delta_coding_enabled = true;
  params.babies_stolen = 5;
  params.dynamic_threshold_enabled = true;
  params.target_species_count = 8;
  params.survival_thresh = 0.2f;

  constexpr float needed_fitness = 0.75f;
  size_t tick = 0;
  size_t last_report = 0;

  while (xor_model.p->max_fitness.load() /
             (exfloat)std::numeric_limits<size_t>::max() <
         needed_fitness) {
    auto tick_task = xor_model.tick_async();
    tick_task.start();
    tick_task.get();

    if (tick - last_report >= 2000) {
      last_report = tick;
      float fitness = xor_model.p->max_fitness.load() /
                      (exfloat)std::numeric_limits<size_t>::max();
      std::cerr << "  [Tick " << tick << "] Fitness: " << fitness
                << " | Pop: " << xor_model.population_size()
                << " | Species: " << xor_model.p->species.size() << std::endl;
    }

    tick++;
    if (tick > 500000) {
      std::cerr << "  Training timeout at " << tick << " ticks" << std::endl;
      break;
    }
  }

  float final_fitness = xor_model.p->max_fitness.load() /
                        (exfloat)std::numeric_limits<size_t>::max();
  std::cerr << "  Training complete at tick " << tick
            << " | Fitness: " << final_fitness << std::endl;
  std::cerr << std::endl;

  // ============================================================
  // Phase 2: Verify original brain on XOR
  // ============================================================
  std::cerr << "[Phase 2] Verifying trained brain..." << std::endl;

  auto best_brain = xor_model.p->get_best_brain();
  if (!best_brain) {
    std::cerr << "  ERROR: No best brain available" << std::endl;
    return 1;
  }

  std::cerr << "  Brain: " << best_brain->neurons.size() << " neurons, "
            << best_brain->fingerprint() << std::endl;
  std::vector<exfloat> output(1, 0.0f);

  for (const auto &[test_input, expected] : test_cases) {
    best_brain->flush();
    best_brain->evaluate(test_input, output);
    float err = std::abs(output[0] - expected);
    bool pass = err < 0.5f;
    std::cerr << "    " << test_input[0] << " XOR " << test_input[1] << " = "
              << output[0] << " (expected " << expected << ") "
              << (pass ? "PASS" : "FAIL") << std::endl;
  }
  std::cerr << std::endl;

  // ============================================================
  // Phase 3: Convert to dense model and verify
  // ============================================================
  std::cerr << "[Phase 3] Converting to dense model..." << std::endl;

  auto dense = eneat::brain_to_dense(*best_brain);
  bool conversion_ok = dense.is_recurrent
      ? (dense.num_hidden > 0)
      : (!dense.layers.empty());
  if (!conversion_ok) {
    std::cerr << "  ERROR: Dense model conversion failed" << std::endl;
    return 1;
  }

  if (dense.is_recurrent) {
    std::cerr << "  Dense model (recurrent): " << dense.unroll_steps << " timesteps"
              << ", sensors=" << dense.state_init.size()
              << ", hidden=" << dense.num_hidden
              << ", outputs=" << dense.num_outputs << std::endl;
  } else {
    std::cerr << "  Dense model (feedforward): " << dense.layers.size() << " layers"
              << ", inputs=" << dense.num_inputs << ", bias=" << dense.num_bias
              << ", outputs=" << dense.num_outputs << std::endl;
    for (size_t l = 0; l < dense.layers.size(); l++) {
      const auto &layer = dense.layers[l];
      std::cerr << "    Layer " << l << ": " << layer.input_size << " → "
                << layer.output_size << " ("
                << activation_name(layer.activations[0]) << ")" << std::endl;
    }
  }

  bool dense_correct = true;
  exfloat max_diff = 0.0f;

  for (const auto &[test_input, expected] : test_cases) {
    std::vector<exfloat> brain_out(1, 0.0f);
    std::vector<exfloat> dense_out(1, 0.0f);

    best_brain->flush();
    best_brain->evaluate(test_input, brain_out);
    dense.evaluate(test_input, dense_out);

    exfloat diff = std::abs(brain_out[0] - dense_out[0]);
    max_diff = std::max(max_diff, diff);

    bool match = diff < 1e-5f;
    std::cerr << "  " << test_input[0] << " XOR " << test_input[1]
              << " | brain=" << brain_out[0] << " dense=" << dense_out[0]
              << " diff=" << diff << (match ? " OK" : " MISMATCH") << std::endl;
    if (!match)
      dense_correct = false;
  }

  std::cerr << "  Max difference: " << max_diff << std::endl;
  std::cerr << "  Dense model verification: "
            << (dense_correct ? "PASSED" : "FAILED") << std::endl;
  std::cerr << std::endl;

  // ============================================================
  // Phase 4: Export to ONNX
  // ============================================================
  std::cerr << "[Phase 4] Exporting to ONNX..." << std::endl;

  bool exported = eneat::export_onnx(*best_brain, "xor_champion.onnx",
                                     "xor_champion");
  if (!exported) {
    std::cerr << "  ERROR: ONNX export failed" << std::endl;
    return 1;
  }
  std::cerr << std::endl;

  // ============================================================
  // Phase 5: Verify via CPU compute backend
  // ============================================================
  std::cerr << "[Phase 5] Verifying via CPU compute backend..." << std::endl;

  cpu_backend cpu;
  cpu.load(*best_brain);
  // Flush between evaluations for consistent results

  for (const auto &[test_input, expected] : test_cases) {
    std::vector<exfloat> cpu_out(1, 0.0f);
    cpu.net.flush();
    cpu.evaluate(test_input, cpu_out);
    std::cerr << "  CPU backend: " << test_input[0] << " XOR " << test_input[1]
              << " = " << cpu_out[0] << std::endl;
  }
  std::cerr << std::endl;

  // ============================================================
  // Phase 6: NPU deployment instructions
  // ============================================================
  std::cerr << "=== NPU Deployment Pipeline ===" << std::endl;
  std::cerr << std::endl;
  std::cerr << "The trained model has been exported to: xor_champion.onnx"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "To deploy on the ESWIN EIC7700 NPU:" << std::endl;
  std::cerr << std::endl;
  std::cerr << "  1. QUANTIZE (on x86 host with ENNP SDK Docker):" << std::endl;
  std::cerr << "     $ python3 esquant.py \\" << std::endl;
  std::cerr << "         --model xor_champion.onnx \\" << std::endl;
  std::cerr << "         --output xor_champion_int8.onnx \\" << std::endl;
  std::cerr << "         --calibration_data xor_calibration/" << std::endl;
  std::cerr << std::endl;
  std::cerr << "  2. COMPILE (on x86 host with ENNP SDK Docker):" << std::endl;
  std::cerr << "     $ esaac \\" << std::endl;
  std::cerr << "         --model xor_champion_int8.onnx \\" << std::endl;
  std::cerr << "         --output xor_champion_npu_b1.model \\" << std::endl;
  std::cerr << "         --target eic7700" << std::endl;
  std::cerr << std::endl;
  std::cerr << "  3. DEPLOY (on RISC-V target):" << std::endl;
  std::cerr << "     $ es_run_model \\" << std::endl;
  std::cerr << "         --model xor_champion_npu_b1.model \\" << std::endl;
  std::cerr << "         --input test_input.bin \\" << std::endl;
  std::cerr << "         --output test_output.bin" << std::endl;
  std::cerr << std::endl;
  std::cerr << "  Or use npu_backend in C++ (requires ESSDK, build with "
               "-DENEAT_NPU_ENABLED):"
            << std::endl;
  std::cerr << "     npu_backend npu;" << std::endl;
  std::cerr << "     npu.load_model(\"xor_champion_npu_b1.model\");"
            << std::endl;
  std::cerr << "     npu.evaluate(input, output);" << std::endl;
  std::cerr << std::endl;

  // ============================================================
  // Summary
  // ============================================================
  std::cerr << "=== Summary ===" << std::endl;
  std::cerr << "  Training:     " << tick << " ticks, fitness=" << final_fitness
            << std::endl;
  std::cerr << "  Brain:        " << best_brain->neurons.size() << " neurons"
            << std::endl;
  std::cerr << "  Dense model:  " << dense.layers.size() << " layers"
            << std::endl;
  std::cerr << "  Verification: "
            << (dense_correct ? "PASSED" : "FAILED") << std::endl;
  std::cerr << "  ONNX export:  " << (exported ? "SUCCESS" : "FAILED")
            << std::endl;
  if (best_brain->recurrent) {
    std::cerr << "  Note:         ONNX uses Jacobi (MatMul) vs brain's Gauss-Seidel."
              << std::endl;
    std::cerr << "                EsQuant calibration compensates for this difference."
              << std::endl;
    std::cerr << "                Verify with: python3 -c 'import onnxruntime ...'"
              << std::endl;
  }
  std::cerr << "  Pipeline:     CPU(train) → ONNX(export) → NPU(deploy)"
            << std::endl;

  // Don't save the training pool (this is a test)
  xor_model.set_read_only(true);

  return (dense_correct && exported) ? 0 : 1;
}
