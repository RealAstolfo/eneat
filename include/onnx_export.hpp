#ifndef ENEAT_ONNX_EXPORT_HPP
#define ENEAT_ONNX_EXPORT_HPP

#include <string>
#include <vector>

#include "brain.hpp"
#include "functions.hpp"
#include "math.hpp"

namespace eneat {

// Dense representation of a brain for verification and export.
// For non-recurrent: sequential feedforward layers.
// For recurrent: single-step weight matrices (W_input, W_recurrent)
//   that are applied iteratively or mapped to ONNX RNN op.
struct dense_model {
  struct layer {
    size_t input_size;
    size_t output_size;
    std::vector<exfloat> weights; // row-major: input_size × output_size
    std::vector<ai_func_type> activations;
  };

  size_t num_inputs;
  size_t num_bias;
  size_t num_outputs;
  bool is_recurrent = false;

  // --- Feedforward mode ---
  std::vector<layer> layers;
  std::vector<size_t> output_positions;
  enum init_type { INIT_INPUT, INIT_BIAS, INIT_ZERO };
  std::vector<init_type> state_init;

  // --- Recurrent mode ---
  size_t num_hidden = 0;        // hidden + output neurons
  size_t unroll_steps = 20;
  // W_input: (num_inputs + num_bias) × num_hidden
  std::vector<exfloat> w_input;
  // W_recurrent: num_hidden × num_hidden
  std::vector<exfloat> w_recurrent;
  // Activation per hidden neuron
  std::vector<ai_func_type> hidden_activations;
  // Which positions in the hidden state are output neurons
  std::vector<size_t> output_hidden_positions;

  void evaluate(const std::vector<exfloat> &input,
                std::vector<exfloat> &output) const;
};

dense_model brain_to_dense(const brain &b, size_t unroll_steps = 20);

bool export_onnx(const brain &b, const std::string &filename,
                 const std::string &model_name = "eneat_model",
                 size_t unroll_steps = 20);

const char *onnx_activation_op(ai_func_type t);

} // namespace eneat

#endif
