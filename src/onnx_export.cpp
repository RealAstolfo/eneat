#include "onnx_export.hpp"
#include "brain.hpp"
#include "functions.hpp"
#include "neuron.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Minimal Protobuf Encoder
// Supports only the subset needed for ONNX ModelProto serialization.
// Wire types: 0=varint, 2=length-delimited, 5=32-bit fixed
// ============================================================================
namespace {

class proto_writer {
  std::vector<uint8_t> buf_;

public:
  const std::vector<uint8_t> &data() const { return buf_; }
  size_t size() const { return buf_.size(); }

  void write_varint(uint64_t v) {
    while (v > 0x7F) {
      buf_.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
      v >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(v & 0x7F));
  }

  void write_tag(uint32_t field, uint32_t wire_type) {
    write_varint((static_cast<uint64_t>(field) << 3) | wire_type);
  }

  // int64/int32/bool/enum → wire type 0 (varint)
  void write_int(uint32_t field, int64_t v) {
    write_tag(field, 0);
    write_varint(static_cast<uint64_t>(v));
  }

  // float → wire type 5 (32-bit fixed)
  void write_float_val(uint32_t field, float v) {
    write_tag(field, 5);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    buf_.push_back(bits & 0xFF);
    buf_.push_back((bits >> 8) & 0xFF);
    buf_.push_back((bits >> 16) & 0xFF);
    buf_.push_back((bits >> 24) & 0xFF);
  }

  // string/bytes → wire type 2 (length-delimited)
  void write_string(uint32_t field, const std::string &s) {
    write_tag(field, 2);
    write_varint(s.size());
    buf_.insert(buf_.end(), s.begin(), s.end());
  }

  // embedded message → wire type 2
  void write_message(uint32_t field, const proto_writer &sub) {
    write_tag(field, 2);
    write_varint(sub.size());
    buf_.insert(buf_.end(), sub.data().begin(), sub.data().end());
  }

  // packed repeated float → wire type 2 (length-delimited block of 4-byte floats)
  void write_packed_floats(uint32_t field, const std::vector<float> &vals) {
    write_tag(field, 2);
    write_varint(vals.size() * 4);
    for (float v : vals) {
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      buf_.push_back(bits & 0xFF);
      buf_.push_back((bits >> 8) & 0xFF);
      buf_.push_back((bits >> 16) & 0xFF);
      buf_.push_back((bits >> 24) & 0xFF);
    }
  }

  // packed repeated int64 → wire type 2 (length-delimited block of varints)
  void write_packed_int64s(uint32_t field,
                           const std::vector<int64_t> &vals) {
    proto_writer tmp;
    for (int64_t v : vals) {
      tmp.write_varint(static_cast<uint64_t>(v));
    }
    write_tag(field, 2);
    write_varint(tmp.size());
    buf_.insert(buf_.end(), tmp.data().begin(), tmp.data().end());
  }
};

// ============================================================================
// ONNX Protobuf Builder Helpers
// Field numbers from onnx.proto3
// ============================================================================

// ONNX DataType enum
constexpr int32_t ONNX_FLOAT = 1;

// ONNX AttributeType enum
constexpr int32_t ATTR_INT = 2;
constexpr int32_t ATTR_FLOAT = 1;

// TensorProto: dims=1, data_type=2, float_data=4(packed), name=8
proto_writer make_tensor(const std::string &name,
                         const std::vector<int64_t> &dims,
                         const std::vector<float> &data) {
  proto_writer t;
  t.write_packed_int64s(1, dims);
  t.write_int(2, ONNX_FLOAT);
  t.write_packed_floats(4, data);
  t.write_string(8, name);
  return t;
}

// TensorShapeProto.Dimension: dim_value=1
proto_writer make_dim(int64_t value) {
  proto_writer d;
  d.write_int(1, value);
  return d;
}

// TensorShapeProto: dim=1(repeated message)
proto_writer make_shape(const std::vector<int64_t> &dims) {
  proto_writer s;
  for (int64_t d : dims) {
    s.write_message(1, make_dim(d));
  }
  return s;
}

// TypeProto.Tensor: elem_type=1, shape=2
proto_writer make_tensor_type(int32_t elem_type,
                              const std::vector<int64_t> &shape) {
  proto_writer tt;
  tt.write_int(1, elem_type);
  tt.write_message(2, make_shape(shape));
  return tt;
}

// TypeProto: tensor_type=1
proto_writer make_type(int32_t elem_type, const std::vector<int64_t> &shape) {
  proto_writer tp;
  tp.write_message(1, make_tensor_type(elem_type, shape));
  return tp;
}

// ValueInfoProto: name=1, type=2
proto_writer make_value_info(const std::string &name,
                             const std::vector<int64_t> &shape) {
  proto_writer vi;
  vi.write_string(1, name);
  vi.write_message(2, make_type(ONNX_FLOAT, shape));
  return vi;
}

// AttributeProto: name=1, type=20, i=3, f=4, ints=8
proto_writer make_attr_int(const std::string &name, int64_t value) {
  proto_writer a;
  a.write_string(1, name);
  a.write_int(20, ATTR_INT);
  a.write_int(3, value);
  return a;
}

proto_writer make_attr_float(const std::string &name, float value) {
  proto_writer a;
  a.write_string(1, name);
  a.write_int(20, ATTR_FLOAT);
  a.write_float_val(4, value);
  return a;
}

// NodeProto: input=1(rep str), output=2(rep str), name=3, op_type=4, attribute=5
proto_writer make_node(const std::string &op_type,
                       const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs,
                       const std::string &name,
                       const std::vector<proto_writer> &attrs = {}) {
  proto_writer n;
  for (const auto &s : inputs)
    n.write_string(1, s);
  for (const auto &s : outputs)
    n.write_string(2, s);
  n.write_string(3, name);
  n.write_string(4, op_type);
  for (const auto &a : attrs)
    n.write_message(5, a);
  return n;
}

// OperatorSetIdProto: domain=1, version=2
proto_writer make_opset(int64_t version,
                        const std::string &domain = "") {
  proto_writer o;
  if (!domain.empty())
    o.write_string(1, domain);
  o.write_int(2, version);
  return o;
}

// ============================================================================
// Topological Analysis
// ============================================================================

struct neuron_layer_info {
  std::vector<int> depths;             // depth per neuron (-1 = unreachable)
  int max_depth;
  // Neurons grouped by depth (depth → [neuron_indices])
  std::vector<std::vector<size_t>> layers;
  // neuron_index → position in accumulated state vector
  std::map<size_t, size_t> neuron_to_state_pos;
};

neuron_layer_info analyze_layers(const brain &b) {
  neuron_layer_info info;
  size_t n = b.neurons.size();
  info.depths.assign(n, -1);

  // Step 1: Backward BFS from outputs to find all reachable neurons.
  // This matches the brain's DFS evaluation which only processes
  // neurons reachable from the output.
  std::vector<bool> reachable(n, false);
  {
    std::vector<size_t> queue;
    for (auto oi : b.output_neurons) {
      reachable[oi] = true;
      queue.push_back(oi);
    }
    for (size_t qi = 0; qi < queue.size(); qi++) {
      for (const auto &conn : b.neurons[queue[qi]].in_connections) {
        // Skip recurrent/time-delayed connections for feedforward export
        if (conn.is_recurrent || conn.is_time_delayed) continue;
        if (!reachable[conn.from_neuron]) {
          reachable[conn.from_neuron] = true;
          queue.push_back(conn.from_neuron);
        }
      }
    }
  }

  // Step 2: Assign depth 0 to reachable input, bias, and orphan neurons
  for (auto i : b.input_neurons)
    if (reachable[i]) info.depths[i] = 0;
  for (auto i : b.bias_neurons)
    if (reachable[i]) info.depths[i] = 0;
  // Orphan neurons: reachable but no feedforward incoming connections.
  // This includes neurons with empty in_connections AND neurons whose
  // connections are all recurrent/time-delayed (filtered out for export).
  for (size_t i = 0; i < n; i++) {
    if (!reachable[i] || info.depths[i] >= 0) continue;
    bool has_feedforward = false;
    for (const auto &conn : b.neurons[i].in_connections) {
      if (!conn.is_recurrent && !conn.is_time_delayed) {
        has_feedforward = true;
        break;
      }
    }
    if (!has_feedforward) {
      info.depths[i] = 0;
    }
  }

  // Step 3: Iteratively compute depths for reachable neurons.
  // Skip unreachable predecessors (they contribute activation(0) ≈ 0 for RELU).
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < n; i++) {
      if (!reachable[i] || info.depths[i] >= 0)
        continue;

      int max_pred = -1;
      bool any_unresolved = false;
      for (const auto &conn : b.neurons[i].in_connections) {
        // Skip recurrent/time-delayed connections for feedforward export
        if (conn.is_recurrent || conn.is_time_delayed) continue;
        if (!reachable[conn.from_neuron]) {
          // Unreachable predecessor: treat as depth 0 constant
          max_pred = std::max(max_pred, 0);
          continue;
        }
        if (info.depths[conn.from_neuron] < 0) {
          any_unresolved = true;
          break;
        }
        max_pred = std::max(max_pred, info.depths[conn.from_neuron]);
      }

      if (!any_unresolved && max_pred >= 0) {
        info.depths[i] = max_pred + 1;
        changed = true;
      }
    }
  }

  // Move all output neurons to a dedicated final layer
  // This ensures the last layer contains exactly the outputs
  int max_hidden_depth = 0;
  for (size_t i = 0; i < n; i++) {
    if (info.depths[i] > max_hidden_depth) {
      bool is_output = false;
      for (auto oi : b.output_neurons) {
        if (oi == i) {
          is_output = true;
          break;
        }
      }
      if (!is_output)
        max_hidden_depth = info.depths[i];
    }
  }

  int output_depth = max_hidden_depth + 1;
  for (auto oi : b.output_neurons) {
    if (info.depths[oi] >= 0) {
      info.depths[oi] = output_depth;
    }
  }
  info.max_depth = output_depth;

  // Group neurons by depth
  info.layers.resize(info.max_depth + 1);
  for (size_t i = 0; i < n; i++) {
    if (info.depths[i] >= 0) {
      info.layers[info.depths[i]].push_back(i);
    }
  }

  // Build neuron → state position mapping
  // State vector layout: [layer0 neurons | layer1 neurons | ... | layerN neurons]
  size_t pos = 0;
  for (int d = 0; d <= info.max_depth; d++) {
    for (auto idx : info.layers[d]) {
      info.neuron_to_state_pos[idx] = pos++;
    }
  }

  return info;
}

// Get the dominant activation function in a set of neurons
ai_func_type dominant_activation(const brain &b,
                                 const std::vector<size_t> &neuron_indices) {
  std::unordered_map<int, size_t> counts;
  for (auto idx : neuron_indices) {
    counts[b.neurons[idx].activation_function]++;
  }
  ai_func_type best = RELU;
  size_t best_count = 0;
  for (auto &[func, count] : counts) {
    if (count > best_count) {
      best_count = count;
      best = static_cast<ai_func_type>(func);
    }
  }
  return best;
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

namespace eneat {

const char *onnx_activation_op(ai_func_type t) {
  switch (t) {
  case RELU:
    return "Relu";
  case LINEAR:
    return nullptr; // identity, no op needed
  case HEAVISIDE:
    return "Relu"; // approximation
  case LOGISTIC:
    return "Sigmoid";
  case SIGMOID:
    return "Tanh"; // bipolar sigmoid (2σ(x)-1) ≈ tanh(x)
  case TANH:
    return "Tanh";
  case GELU:
    return "Relu"; // approximation for NPU compatibility
  case SWISH:
    return "Sigmoid"; // approximation
  case LEAKY_RELU:
    return "LeakyRelu";
  case NORMALIZE:
    return nullptr; // handled as identity for NPU
  default:
    return "Relu";
  }
}

// ============================================================================
// brain_to_dense: Convert brain topology to dense layered representation
// ============================================================================

dense_model brain_to_dense(const brain &b, size_t unroll_steps) {
  dense_model model;
  model.num_inputs = b.input_neurons.size();
  model.num_bias = b.bias_neurons.size();
  model.num_outputs = b.output_neurons.size();
  model.is_recurrent = b.recurrent;

  if (b.recurrent) {
    // ============ RECURRENT MODE ============
    // Build W_input and W_recurrent matrices that match evaluate_recurrent().
    // Sensor neurons (input + bias) form the "input" vector.
    // All other neurons form the "hidden state" vector.
    model.unroll_steps = unroll_steps;

    // Map neuron indices to positions in input vector or hidden state
    std::set<size_t> sensor_set;
    for (auto i : b.input_neurons) sensor_set.insert(i);
    for (auto i : b.bias_neurons) sensor_set.insert(i);

    // Build ordered lists: sensors and hidden neurons
    std::vector<size_t> sensor_indices;
    std::vector<size_t> hidden_indices;
    for (size_t i = 0; i < b.neurons.size(); i++) {
      if (sensor_set.count(i)) {
        sensor_indices.push_back(i);
      } else {
        hidden_indices.push_back(i);
      }
    }

    size_t S = sensor_indices.size();  // inputs + biases
    size_t H = hidden_indices.size();  // hidden + output
    model.num_hidden = H;

    // Neuron index → position in sensor or hidden vector
    std::map<size_t, size_t> sensor_pos, hidden_pos;
    for (size_t i = 0; i < S; i++) sensor_pos[sensor_indices[i]] = i;
    for (size_t i = 0; i < H; i++) hidden_pos[hidden_indices[i]] = i;

    // W_input: S × H (sensor → hidden weights)
    model.w_input.resize(S * H, 0.0f);
    // W_recurrent: H × H (hidden → hidden weights, including recurrent)
    model.w_recurrent.resize(H * H, 0.0f);
    // Activation per hidden neuron + track which have connections
    model.hidden_activations.resize(H);
    std::vector<bool> has_connections(H, false);

    for (size_t j = 0; j < H; j++) {
      size_t nidx = hidden_indices[j];
      model.hidden_activations[j] = b.neurons[nidx].activation_function;

      for (const auto &conn : b.neurons[nidx].in_connections) {
        if (conn.is_time_delayed) continue;

        has_connections[j] = true;
        if (sensor_pos.count(conn.from_neuron)) {
          model.w_input[sensor_pos[conn.from_neuron] * H + j] = conn.weight;
        } else if (hidden_pos.count(conn.from_neuron)) {
          model.w_recurrent[hidden_pos[conn.from_neuron] * H + j] = conn.weight;
        }
      }
      // Mark neurons with no connections as LINEAR (identity)
      // so activation(0) = 0, matching brain's skip behavior
      if (!has_connections[j]) {
        model.hidden_activations[j] = LINEAR;
      }
    }

    // Build state init for sensors
    for (auto idx : sensor_indices) {
      if (b.neurons[idx].type == INPUT)
        model.state_init.push_back(dense_model::INIT_INPUT);
      else if (b.neurons[idx].type == BIAS)
        model.state_init.push_back(dense_model::INIT_BIAS);
      else
        model.state_init.push_back(dense_model::INIT_ZERO);
    }

    // Find output positions in hidden state
    std::set<size_t> output_set(b.output_neurons.begin(), b.output_neurons.end());
    for (size_t j = 0; j < H; j++) {
      if (output_set.count(hidden_indices[j])) {
        model.output_hidden_positions.push_back(j);
      }
    }
  } else {
    // ============ FEEDFORWARD MODE ============
    auto info = analyze_layers(b);

    std::set<size_t> input_set(b.input_neurons.begin(), b.input_neurons.end());
    std::set<size_t> bias_set(b.bias_neurons.begin(), b.bias_neurons.end());
    for (auto idx : info.layers[0]) {
      if (input_set.count(idx))
        model.state_init.push_back(dense_model::INIT_INPUT);
      else if (bias_set.count(idx))
        model.state_init.push_back(dense_model::INIT_BIAS);
      else
        model.state_init.push_back(dense_model::INIT_ZERO);
    }

    for (int d = 1; d <= info.max_depth; d++) {
      const auto &layer_neurons = info.layers[d];
      if (layer_neurons.empty()) continue;

      size_t input_size = 0;
      for (int prev_d = 0; prev_d < d; prev_d++)
        input_size += info.layers[prev_d].size();

      size_t output_size = layer_neurons.size();
      dense_model::layer layer;
      layer.input_size = input_size;
      layer.output_size = output_size;
      layer.weights.resize(input_size * output_size, 0.0f);
      layer.activations.resize(output_size);

      for (size_t j = 0; j < output_size; j++) {
        size_t neuron_idx = layer_neurons[j];
        layer.activations[j] = b.neurons[neuron_idx].activation_function;
        for (const auto &conn : b.neurons[neuron_idx].in_connections) {
          if (conn.is_recurrent || conn.is_time_delayed) continue;
          auto it = info.neuron_to_state_pos.find(conn.from_neuron);
          if (it != info.neuron_to_state_pos.end() && it->second < input_size)
            layer.weights[it->second * output_size + j] = conn.weight;
        }
      }
      model.layers.push_back(std::move(layer));
    }

    for (auto oi : b.output_neurons) {
      auto it = info.neuron_to_state_pos.find(oi);
      if (it != info.neuron_to_state_pos.end())
        model.output_positions.push_back(it->second);
    }
  }

  return model;
}

// ============================================================================
// dense_model::evaluate: Reference CPU evaluation of the dense model
// ============================================================================

static exfloat apply_activation(ai_func_type func, exfloat sum) {
  switch (func) {
  case RELU: return relu(sum);
  case LINEAR: return linear(sum);
  case HEAVISIDE: return heaviside(sum);
  case LOGISTIC: return logistic(sum);
  case SIGMOID: return sigmoid(sum);
  case TANH: return tanh(sum);
  case GELU: return gelu(sum);
  case SWISH: return swish(sum);
  case LEAKY_RELU: return leaky_relu(sum);
  case NORMALIZE: return normalize(sum);
  default: return relu(sum);
  }
}

void dense_model::evaluate(const std::vector<exfloat> &input,
                           std::vector<exfloat> &output) const {
  if (is_recurrent) {
    // ============ RECURRENT EVALUATION ============
    // Matches brain::evaluate_recurrent():
    //   For each timestep: H = activation(sensors @ W_input + H_prev @ W_recurrent)
    size_t S = state_init.size();
    size_t H = num_hidden;

    // Build sensor vector
    std::vector<exfloat> sensors(S, 0.0f);
    size_t input_idx = 0;
    for (size_t i = 0; i < S; i++) {
      switch (state_init[i]) {
      case INIT_INPUT:
        sensors[i] = input_idx < input.size() ? input[input_idx++] : 0.0f;
        break;
      case INIT_BIAS: sensors[i] = 1.0f; break;
      case INIT_ZERO: sensors[i] = 0.0f; break;
      }
    }

    // Hidden state (initialized to 0)
    std::vector<exfloat> h(H, 0.0f);

    // Match brain::evaluate_recurrent(): iterate until all outputs are activated,
    // max unroll_steps iterations. The brain exits when outputs_ready() and at
    // least one iteration has completed. Uses Gauss-Seidel (in-place) updates.
    // Match brain::evaluate_recurrent loop:
    //   while (!outputs_ready() || !onetime) { process; onetime=true; }
    // Check BEFORE each iteration (matching the while condition).
    std::vector<bool> activated(H, false);
    bool onetime = false;
    for (size_t t = 0; t < unroll_steps; t++) {
      // Check exit: outputs ready AND at least one pass done
      if (onetime) {
        bool all_ready = true;
        for (auto pos : output_hidden_positions)
          if (!activated[pos]) { all_ready = false; break; }
        if (all_ready) break;
      }

      for (size_t j = 0; j < H; j++) {
        // Skip neurons with no connections (matches brain's skip)
        bool has_weights = false;
        for (size_t i = 0; i < S && !has_weights; i++)
          if (w_input[i * H + j] != 0.0f) has_weights = true;
        for (size_t i = 0; i < H && !has_weights; i++)
          if (w_recurrent[i * H + j] != 0.0f) has_weights = true;
        if (!has_weights) continue;

        exfloat sum = 0.0f;
        for (size_t i = 0; i < S; i++)
          sum += sensors[i] * w_input[i * H + j];
        for (size_t i = 0; i < H; i++)
          sum += h[i] * w_recurrent[i * H + j];
        h[j] = apply_activation(hidden_activations[j], sum);
        activated[j] = true;
      }
      onetime = true;
    }

    // Extract outputs from hidden state
    for (size_t i = 0; i < output_hidden_positions.size() && i < output.size(); i++)
      output[i] = h[output_hidden_positions[i]];

  } else {
    // ============ FEEDFORWARD EVALUATION ============
    std::vector<exfloat> state;
    state.reserve(state_init.size());
    size_t input_idx = 0;
    for (auto init : state_init) {
      switch (init) {
      case INIT_INPUT:
        state.push_back(input_idx < input.size() ? input[input_idx++] : 0.0f);
        break;
      case INIT_BIAS: state.push_back(1.0f); break;
      case INIT_ZERO: state.push_back(0.0f); break;
      }
    }

    for (const auto &layer : layers) {
      std::vector<exfloat> next(layer.output_size, 0.0f);
      for (size_t j = 0; j < layer.output_size; j++) {
        exfloat sum = 0.0f;
        for (size_t i = 0; i < layer.input_size && i < state.size(); i++)
          sum += state[i] * layer.weights[i * layer.output_size + j];
        next[j] = apply_activation(layer.activations[j], sum);
      }
      state.insert(state.end(), next.begin(), next.end());
    }

    for (size_t i = 0; i < output_positions.size() && i < output.size(); i++)
      if (output_positions[i] < state.size())
        output[i] = state[output_positions[i]];
  }
}

// ============================================================================
// export_onnx: Write brain as ONNX protobuf file
// ============================================================================

bool export_onnx(const brain &b, const std::string &filename,
                 const std::string &model_name,
                 size_t unroll_steps) {
  // Build the dense model first (handles both recurrent and feedforward)
  // For recurrent: determine actual iteration count by probing.
  // The brain exits early when outputs are activated. The ONNX graph must
  // use the SAME number of iterations to produce matching results.
  size_t actual_steps = unroll_steps;
  if (b.recurrent) {
    // Probe with zero input to find how many iterations the brain uses
    std::vector<bool> activated_probe(b.neurons.size(), false);
    size_t steps = 0;
    for (size_t t = 0; t < unroll_steps; t++) {
      bool any_computed = false;
      for (size_t i = 0; i < b.neurons.size(); i++) {
        if (b.neurons[i].type == INPUT || b.neurons[i].type == BIAS) continue;
        if (b.neurons[i].in_connections.empty()) continue;
        activated_probe[i] = true;
        any_computed = true;
      }
      steps = t + 1;
      if (any_computed && t > 0) {
        bool all_outputs = true;
        for (auto oi : b.output_neurons) {
          if (!activated_probe[oi]) { all_outputs = false; break; }
        }
        if (all_outputs) break;
      }
    }
    actual_steps = steps;
  }
  auto dm = brain_to_dense(b, actual_steps);

  size_t num_inputs = dm.num_inputs;
  size_t num_outputs = dm.num_outputs;

  proto_writer graph;
  graph.write_string(2, model_name);
  graph.write_message(11, make_value_info("X", {1, (int64_t)num_inputs}));
  graph.write_message(12, make_value_info("Y", {1, (int64_t)num_outputs}));

  if (dm.is_recurrent) {
    // ============ RECURRENT ONNX EXPORT ============
    // Unroll into sequential MatMul steps:
    //   For each timestep t: h_t = activation(sensors @ W_in + h_{t-1} @ W_rec)
    // This is NPU-friendly: just repeated MatMul + Add + Activation.
    size_t S = dm.state_init.size(); // sensors
    size_t H = dm.num_hidden;

    // Build sensor vector from input + bias constant
    size_t bias_count = 0;
    for (auto init : dm.state_init)
      if (init == dense_model::INIT_BIAS) bias_count++;

    if (bias_count > 0) {
      std::vector<float> bias_vals(bias_count, 1.0f);
      graph.write_message(5, make_tensor("bias_const", {1, (int64_t)bias_count}, bias_vals));

      graph.write_message(1, make_node("Concat", {"X", "bias_const"}, {"sensors"},
                                       "concat_sensors", {make_attr_int("axis", 1)}));
    } else {
      graph.write_message(1, make_node("Identity", {"X"}, {"sensors"}, "id_sensors"));
    }

    // W_in and W_rec as initializers (shared across all timesteps)
    std::vector<float> w_in_f(dm.w_input.begin(), dm.w_input.end());
    std::vector<float> w_rec_f(dm.w_recurrent.begin(), dm.w_recurrent.end());
    graph.write_message(5, make_tensor("W_in", {(int64_t)S, (int64_t)H}, w_in_f));

    graph.write_message(5, make_tensor("W_rec", {(int64_t)H, (int64_t)H}, w_rec_f));


    // Initial hidden state = zeros
    std::vector<float> h_init(H, 0.0f);
    graph.write_message(5, make_tensor("h_0", {1, (int64_t)H}, h_init));


    // Determine dominant activation for the hidden neurons
    ai_func_type dom_act = RELU;
    {
      std::unordered_map<int, size_t> counts;
      for (auto a : dm.hidden_activations) counts[a]++;
      size_t best = 0;
      for (auto &[func, count] : counts) {
        if (count > best) { best = count; dom_act = static_cast<ai_func_type>(func); }
      }
    }
    const char *act_op = onnx_activation_op(dom_act);

    // Unrolled timesteps
    std::string h_prev = "h_0";
    for (size_t t = 0; t < dm.unroll_steps; t++) {
      std::string ts = std::to_string(t);
      std::string inp_part = "inp_" + ts;
      std::string rec_part = "rec_" + ts;
      std::string sum_name = "sum_" + ts;
      std::string h_name = "h_" + std::to_string(t + 1);

      // sensors @ W_in
      graph.write_message(1, make_node("MatMul", {"sensors", "W_in"}, {inp_part},
                                       "matmul_in_" + ts));
      // h_prev @ W_rec
      graph.write_message(1, make_node("MatMul", {h_prev, "W_rec"}, {rec_part},
                                       "matmul_rec_" + ts));
      // Add
      graph.write_message(1, make_node("Add", {inp_part, rec_part}, {sum_name},
                                       "add_" + ts));
      // Activation
      if (act_op) {
        std::vector<proto_writer> attrs;
        if (dom_act == LEAKY_RELU) attrs.push_back(make_attr_float("alpha", 0.01f));
        graph.write_message(1, make_node(act_op, {sum_name}, {h_name},
                                         "act_" + ts, attrs));
      } else {
        graph.write_message(1, make_node("Identity", {sum_name}, {h_name}, "id_" + ts));
      }
      h_prev = h_name;
    }

    // Extract output neurons from final hidden state
    // If all outputs are contiguous at the end, just slice. Otherwise use Gather.
    // For simplicity, build a selection matrix: H × num_outputs
    std::vector<float> sel(H * num_outputs, 0.0f);
    for (size_t i = 0; i < dm.output_hidden_positions.size() && i < num_outputs; i++) {
      sel[dm.output_hidden_positions[i] * num_outputs + i] = 1.0f;
    }
    graph.write_message(5, make_tensor("W_sel", {(int64_t)H, (int64_t)num_outputs}, sel));

    graph.write_message(1, make_node("MatMul", {h_prev, "W_sel"}, {"Y"}, "select_output"));

    std::cerr << "Exported recurrent ONNX model to " << filename << std::endl;
    std::cerr << "  Inputs: " << num_inputs << ", Hidden: " << H
              << ", Outputs: " << num_outputs
              << ", Unrolled steps: " << dm.unroll_steps << std::endl;

  } else {
    // ============ FEEDFORWARD ONNX EXPORT ============
    auto info = analyze_layers(b);

    size_t num_bias = dm.num_bias;
    if (num_bias > 0) {
      std::vector<float> bias_vals(num_bias, 1.0f);
      graph.write_message(5, make_tensor("bias_const", {1, (int64_t)num_bias}, bias_vals));
      graph.write_message(11, make_value_info("bias_const", {1, (int64_t)num_bias}));
      graph.write_message(1, make_node("Concat", {"X", "bias_const"}, {"state_0"},
                                       "concat_bias", {make_attr_int("axis", 1)}));
    } else {
      graph.write_message(1, make_node("Identity", {"X"}, {"state_0"}, "identity_input"));
    }

    std::string prev_state = "state_0";
    size_t state_size = info.layers[0].size();
    int layer_idx = 0;

    for (int d = 1; d <= info.max_depth; d++) {
      const auto &layer_neurons = info.layers[d];
      if (layer_neurons.empty()) continue;

      size_t layer_size = layer_neurons.size();
      std::string w_name = "W_" + std::to_string(layer_idx);
      std::string pre_name = "pre_" + std::to_string(layer_idx);
      std::string act_name = "act_" + std::to_string(layer_idx);
      std::string state_name = "state_" + std::to_string(layer_idx + 1);

      std::vector<float> weights(state_size * layer_size, 0.0f);
      for (size_t j = 0; j < layer_size; j++) {
        size_t neuron_idx = layer_neurons[j];
        for (const auto &conn : b.neurons[neuron_idx].in_connections) {
          if (conn.is_recurrent || conn.is_time_delayed) continue;
          auto it = info.neuron_to_state_pos.find(conn.from_neuron);
          if (it != info.neuron_to_state_pos.end() && it->second < state_size)
            weights[it->second * layer_size + j] = conn.weight;
        }
      }

      graph.write_message(5, make_tensor(w_name, {(int64_t)state_size, (int64_t)layer_size}, weights));
      // Initializers don't need to be in graph inputs (opset 13+)
      graph.write_message(1, make_node("MatMul", {prev_state, w_name}, {pre_name},
                                       "matmul_" + std::to_string(layer_idx)));

      ai_func_type layer_act = dominant_activation(b, layer_neurons);
      const char *act_op_name = onnx_activation_op(layer_act);

      std::string layer_output;
      if (act_op_name) {
        std::vector<proto_writer> attrs;
        if (layer_act == LEAKY_RELU) attrs.push_back(make_attr_float("alpha", 0.01f));
        graph.write_message(1, make_node(act_op_name, {pre_name}, {act_name},
                                         "act_" + std::to_string(layer_idx), attrs));
        layer_output = act_name;
      } else {
        layer_output = pre_name;
      }

      if (d == info.max_depth) {
        graph.write_message(1, make_node("Identity", {layer_output}, {"Y"}, "output_identity"));
      } else {
        graph.write_message(1, make_node("Concat", {prev_state, layer_output}, {state_name},
                                         "concat_" + std::to_string(layer_idx),
                                         {make_attr_int("axis", 1)}));
        prev_state = state_name;
        state_size += layer_size;
      }
      layer_idx++;
    }

    if (layer_idx == 0) {
      std::cerr << "export_onnx: no compute layers found" << std::endl;
      return false;
    }

    std::cerr << "Exported feedforward ONNX model to " << filename << std::endl;
    std::cerr << "  Inputs: " << num_inputs << ", Outputs: " << num_outputs
              << ", Layers: " << layer_idx << std::endl;
  }

  std::cerr << "  Total neurons: " << b.neurons.size() << std::endl;

  // ---- Write ONNX model ----
  proto_writer onnx_model;
  onnx_model.write_int(1, 9);
  onnx_model.write_message(8, make_opset(18));
  onnx_model.write_string(2, "eneat");
  onnx_model.write_string(3, "1.0.0");
  onnx_model.write_message(7, graph);

  std::ofstream f(filename, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "export_onnx: cannot open " << filename << std::endl;
    return false;
  }
  const auto &data = onnx_model.data();
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
  f.close();
  return true;
}

} // namespace eneat
