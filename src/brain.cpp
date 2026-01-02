#include "brain.hpp"
#include "functions.hpp"
#include "gene.hpp"
#include "neuron.hpp"
#include "trait.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <ostream>
#include <stddef.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

std::istream &operator>>(std::istream &input, brain &b) {
  b.neurons.clear();
  b.input_neurons.clear();
  b.output_neurons.clear();
  b.traits.clear();
  input >> b.recurrent;
  input >> b.hebbian_enabled;

  // Read traits
  size_t trait_count;
  input >> trait_count;
  b.traits.resize(trait_count);
  for (size_t i = 0; i < trait_count; i++) {
    input >> b.traits[i];
  }

  size_t neuron_number;
  input >> neuron_number;
  b.neurons.resize(neuron_number);
  for (size_t i = 0; i < neuron_number; i++) {
    size_t tmp_type;
    neuron_place type;
    size_t tmp_act;
    ai_func_type act;
    size_t input_size;
    b.neurons[i].value = 0.0f;
    b.neurons[i].last_activation = 0.0f;
    b.neurons[i].last_activation2 = 0.0f;
    b.neurons[i].visited = false;
    input >> tmp_type;
    type = (neuron_place)tmp_type;
    input >> tmp_act;
    act = (ai_func_type)tmp_act;
    input >> b.neurons[i].trait_id;
    switch (type) {
    case INPUT:
      b.input_neurons.push_back(i);
      break;
    case OUTPUT:
      b.output_neurons.push_back(i);
      break;
    case BIAS:
      b.bias_neurons.push_back(i);
      break;
    case HIDDEN:
      break;
    }

    b.neurons[i].type = type;
    b.neurons[i].activation_function = act;
    input >> input_size;
    for (size_t j = 0; j < input_size; j++) {
      size_t from;
      exfloat w;
      size_t tid;
      bool rec;
      bool td;
      input >> from >> w >> tid >> rec >> td;
      b.neurons[i].in_connections.emplace_back(from, w, tid, rec, td);
    }
  }
  return input;
}

std::ostream &operator<<(std::ostream &output, brain &b) {
  output << b.recurrent << std::endl;
  output << b.hebbian_enabled << std::endl;

  // Write traits
  output << b.traits.size() << std::endl;
  for (size_t i = 0; i < b.traits.size(); i++) {
    output << b.traits[i] << std::endl;
  }

  output << b.neurons.size() << std::endl << std::endl;
  for (size_t i = 0; i < b.neurons.size(); i++) {
    output << (size_t)b.neurons[i].type << " ";
    output << (size_t)b.neurons[i].activation_function << " ";
    output << b.neurons[i].trait_id << " ";
    output << b.neurons[i].in_connections.size() << std::endl;
    for (size_t j = 0; j < b.neurons[i].in_connections.size(); j++) {
      const auto& conn = b.neurons[i].in_connections[j];
      output << conn.from_neuron << " " << conn.weight << " "
             << conn.trait_id << " " << conn.is_recurrent << " "
             << conn.is_time_delayed << " ";
    }
    output << std::endl << std::endl;
  }
  return output;
}

void brain::operator=(const genome &g) {
  const size_t &input_size = g.network_info.input_size;
  const size_t &output_size = g.network_info.output_size;
  const size_t &bias_size = g.network_info.bias_size;
  recurrent = g.network_info.recurrent;
  neurons.clear();
  input_neurons.clear();
  bias_neurons.clear();
  output_neurons.clear();
  traits.clear();

  // Copy traits from genome
  traits = g.traits;
  hebbian_enabled = !traits.empty();

  // Build a set of evolved bias neuron IDs from genes with is_bias_source flag
  std::unordered_set<size_t> evolved_bias_set;
  for (const auto &[_, gene] : g.genes) {
    if (gene.is_bias_source) {
      evolved_bias_set.insert(gene.from_node);
    }
  }

  neuron tmp;
  for (size_t i = 0; i < input_size; i++) {
    neurons.push_back(tmp);
    neurons.back().type = INPUT;
    input_neurons.push_back(neurons.size() - 1);
  }
  for (size_t i = 0; i < bias_size; i++) {
    neurons.push_back(tmp);
    neurons.back().type = BIAS;
    bias_neurons.push_back(neurons.size() - 1);
  }
  for (size_t i = 0; i < output_size; i++) {
    neurons.push_back(tmp);
    neurons.back().type = OUTPUT;
    output_neurons.push_back(neurons.size() - 1);
  }

  std::map<size_t, size_t> table;
  for (size_t i = 0;
       i < input_neurons.size() + output_neurons.size() + bias_neurons.size();
       i++)
    table[i] = i;
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (!it->second.enabled)
      continue;
    neuron n;
    n.activation_function = it->second.activation;

    // Check if from_node is an evolved bias neuron
    if (table.find(it->second.from_node) == table.end()) {
      if (evolved_bias_set.count(it->second.from_node)) {
        n.type = BIAS;
        neurons.push_back(n);
        size_t idx = neurons.size() - 1;
        table[it->second.from_node] = idx;
        bias_neurons.push_back(idx);
      } else {
        neurons.push_back(n);
        table[it->second.from_node] = neurons.size() - 1;
      }
    }

    // Check if to_node is an evolved bias neuron (unlikely but handle it)
    if (table.find(it->second.to_node) == table.end()) {
      if (evolved_bias_set.count(it->second.to_node)) {
        n.type = BIAS;
        neurons.push_back(n);
        size_t idx = neurons.size() - 1;
        table[it->second.to_node] = idx;
        bias_neurons.push_back(idx);
      } else {
        neurons.push_back(n);
        table[it->second.to_node] = neurons.size() - 1;
      }
    }
  }

  // Add connections with trait information and derive trait params
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (!it->second.enabled)
      continue;

    neuron_connection conn(
        table[it->second.from_node],
        it->second.weight,
        it->second.trait_id,
        it->second.is_recurrent,
        it->second.is_time_delayed
    );

    // Derive trait parameters for the connection
    if (it->second.trait_id > 0 && it->second.trait_id <= g.traits.size()) {
      const auto& trait = g.traits[it->second.trait_id - 1];
      for (size_t p = 0; p < neuron_connection::NUM_TRAIT_PARAMS && p < eneat::NUM_TRAIT_PARAMS; p++) {
        conn.params[p] = trait.params[p];
      }
    }

    neurons[table[it->second.to_node]].in_connections.push_back(conn);
  }

  // Derive trait parameters for neurons (node traits)
  for (const auto& [node_id, trait_id] : g.node_traits) {
    if (table.find(node_id) != table.end() && trait_id > 0 && trait_id <= g.traits.size()) {
      size_t neuron_idx = table[node_id];
      neurons[neuron_idx].trait_id = trait_id;
      const auto& trait = g.traits[trait_id - 1];
      for (size_t p = 0; p < neuron::NUM_TRAIT_PARAMS && p < eneat::NUM_TRAIT_PARAMS; p++) {
        neurons[neuron_idx].params[p] = trait.params[p];
      }
    }
  }
}

void brain::evaluate_nonrecurrent(const std::vector<exfloat> &input,
                                  std::vector<exfloat> &output) {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].value = 0.0f;
    neurons[i].visited = false;
  }

  // Input neurons are "dead ends" when it comes to reverse walking. so make it
  // default visited.
  for (size_t i = 0; i < input.size() && i < input_neurons.size(); i++) {
    neurons[input_neurons[i]].value = input[i];
    neurons[input_neurons[i]].visited = true;
  }

  // Bias neurons are "dead ends" when it comes to reverse walking. so make it
  // default visited.
  for (size_t i = 0; i < bias_neurons.size(); i++) {
    neurons[bias_neurons[i]].value = 1.0f;
    neurons[bias_neurons[i]].visited = true;
  }

  std::stack<size_t> s;
  for (size_t i = 0; i < output_neurons.size(); i++)
    s.push(output_neurons[i]);

  while (!s.empty()) {
    size_t t = s.top();
    if (neurons[t].visited) {
      exfloat sum = 0.0f;
      for (size_t i = 0; i < neurons[t].in_connections.size(); i++)
        sum += neurons[neurons[t].in_connections[i].from_neuron].value *
               neurons[t].in_connections[i].weight;
      switch (neurons[t].activation_function) {
      case RELU:
        neurons[t].value = relu(sum);
        break;
      case LINEAR:
        neurons[t].value = linear(sum);
        break;
      case HEAVISIDE:
        neurons[t].value = heaviside(sum);
        break;
      case LOGISTIC:
        neurons[t].value = logistic(sum);
        break;
      case SIGMOID:
        neurons[t].value = sigmoid(sum);
        break;
      case TANH:
        neurons[t].value = tanh(sum);
        break;
      case GELU:
        neurons[t].value = gelu(sum);
        break;
      case SWISH:
        neurons[t].value = swish(sum);
        break;
      case LEAKY_RELU:
        neurons[t].value = leaky_relu(sum, neurons[t].value);
        break;
      case NORMALIZE:
        neurons[t].value = normalize(sum, neurons[t].value);
        break;
      }
      s.pop();
    } else {
      // Wherever we are, walk towards the start of the network
      neurons[t].visited = true;
      for (size_t i = 0; i < neurons[t].in_connections.size(); i++)
        if (!neurons[neurons[t].in_connections[i].from_neuron].visited)
          s.push(neurons[t].in_connections[i].from_neuron);
    }
  }

  for (size_t i = 0; i < output_neurons.size() && i < output.size(); i++)
    output[i] = neurons[output_neurons[i]].value;
}

void brain::evaluate_recurrent(const std::vector<exfloat> &input,
                               std::vector<exfloat> &output) {
  // Shift activation history: last_activation2 = last_activation, last_activation = current
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].last_activation2 = neurons[i].last_activation;
    neurons[i].last_activation = neurons[i].value;
  }

  for (size_t i = 0; i < input.size() && i < input_neurons.size(); i++) {
    neurons[input_neurons[i]].value = input[i];
    neurons[input_neurons[i]].visited = true;
  }

  for (size_t i = 0; i < bias_neurons.size(); i++) {
    neurons[bias_neurons[i]].value = 1.0f;
    neurons[bias_neurons[i]].visited = true;
  }

  for (size_t i = 0; i < neurons.size(); i++) {
    exfloat sum = 0.0f;
    for (size_t j = 0; j < neurons[i].in_connections.size(); j++) {
      const auto& conn = neurons[i].in_connections[j];
      exfloat input_activation;

      // Time-delayed connections use last_activation (t-1 for normal, t-2 for time-delayed)
      // rtNEAT's get_active_out_td() returns last_activation if activation_count > 1
      if (conn.is_time_delayed) {
        // Time-delayed: use activation from two timesteps ago
        input_activation = (neurons[conn.from_neuron].activation_count > 1)
            ? neurons[conn.from_neuron].last_activation
            : 0.0f;
      } else {
        // Normal connection: use current value
        input_activation = neurons[conn.from_neuron].value;
      }

      sum += input_activation * conn.weight;
    }
    if (neurons[i].in_connections.size() > 0) {
      switch (neurons[i].activation_function) {
      case RELU:
        neurons[i].value = relu(sum);
        break;
      case LINEAR:
        neurons[i].value = linear(sum);
        break;
      case HEAVISIDE:
        neurons[i].value = heaviside(sum);
        break;
      case LOGISTIC:
        neurons[i].value = logistic(sum);
        break;
      case SIGMOID:
        neurons[i].value = sigmoid(sum);
        break;
      case TANH:
        neurons[i].value = tanh(sum);
        break;
      case GELU:
        neurons[i].value = gelu(sum);
        break;
      case SWISH:
        neurons[i].value = swish(sum);
        break;
      case LEAKY_RELU:
        neurons[i].value = leaky_relu(sum, neurons[i].value);
        break;
      case NORMALIZE:
        neurons[i].value = normalize(sum, neurons[i].value);
        break;
      }
    }
    neurons[i].activation_count += 1.0f;
  }

  // Apply Hebbian learning if enabled
  if (hebbian_enabled) {
    apply_hebbian_learning();
  }

  for (size_t i = 0; i < output_neurons.size() && i < output.size(); i++)
    output[i] = neurons[output_neurons[i]].value;
}

// rtNEAT oldhebbian algorithm
// Based on rtNEAT neat.cpp:463
exfloat brain::oldhebbian(exfloat weight, exfloat maxweight, exfloat active_in,
                          exfloat active_out, exfloat hebb_rate,
                          exfloat pre_rate, exfloat post_rate) {
  bool neg = false;
  exfloat delta;

  if (maxweight < 5.0f) maxweight = 5.0f;
  if (weight > maxweight) weight = maxweight;
  if (weight < -maxweight) weight = -maxweight;

  if (weight < 0) {
    neg = true;
    weight = -weight;
  }

  if (!neg) {
    // Excitatory synapse
    delta = hebb_rate * (maxweight - weight) * active_in * active_out +
            pre_rate * weight * active_in * (active_out - 1.0f) +
            post_rate * weight * (active_in - 1.0f) * active_out;

    if (weight + delta > 0)
      return weight + delta;
    return 0.01f;
  } else {
    // Inhibitory synapse: strengthen when output is low and input is high
    delta = hebb_rate * (maxweight - weight) * active_in * (1.0f - active_out) +
            -5.0f * hebb_rate * weight * active_in * active_out;

    if (-(weight + delta) < 0)
      return -(weight + delta);
    return -0.01f;
  }
}

// rtNEAT hebbian algorithm (Floreano & Urzelai 2000)
// Based on rtNEAT neat.cpp:534
exfloat brain::hebbian(exfloat weight, exfloat maxweight, exfloat active_in,
                       exfloat active_out, exfloat hebb_rate,
                       exfloat pre_rate, exfloat post_rate) {
  (void)post_rate;  // Not used in this variant (matches rtNEAT reference)

  bool neg = false;
  exfloat delta;
  exfloat topweight;

  if (maxweight < 5.0f) maxweight = 5.0f;
  if (weight > maxweight) weight = maxweight;
  if (weight < -maxweight) weight = -maxweight;

  if (weight < 0) {
    neg = true;
    weight = -weight;
  }

  topweight = weight + 2.0f;
  if (topweight > maxweight) topweight = maxweight;

  if (!neg) {
    // Excitatory synapse
    delta = hebb_rate * (maxweight - weight) * active_in * active_out +
            pre_rate * topweight * active_in * (active_out - 1.0f);

    return weight + delta;
  } else {
    // Inhibitory synapse
    delta = pre_rate * (maxweight - weight) * active_in * (1.0f - active_out) +
            -hebb_rate * (topweight + 2.0f) * active_in * active_out;

    return -(weight + delta);
  }
}

// Apply Hebbian learning using rtNEAT algorithms
// Trait params:
// params[0] = Hebbian learning rate (hebb_rate)
// params[1] = Presynaptic learning rate (pre_rate)
// params[2] = Postsynaptic learning rate (post_rate)
// params[3] = Max weight limit (maxweight)
// params[4] = Learning algorithm: 0 = hebbian, 1 = oldhebbian
void brain::apply_hebbian_learning() {
  for (size_t i = 0; i < neurons.size(); i++) {
    exfloat post_activation = neurons[i].value;

    for (size_t j = 0; j < neurons[i].in_connections.size(); j++) {
      auto& conn = neurons[i].in_connections[j];

      // Skip if no trait assigned
      if (conn.trait_id == 0 || conn.trait_id > traits.size())
        continue;

      const auto& t = traits[conn.trait_id - 1];  // trait_id is 1-indexed

      // Skip if learning not enabled for this trait
      if (!t.is_learning_enabled())
        continue;

      exfloat pre_activation = neurons[conn.from_neuron].last_activation;

      // Get learning parameters from trait
      exfloat hebb_rate = t.params[0];
      exfloat pre_rate = t.params[1];
      exfloat post_rate = t.params[2];
      exfloat maxweight = t.params[3];
      bool use_oldhebbian = (t.params[4] > 0.5f);

      // Apply the appropriate Hebbian learning algorithm
      if (use_oldhebbian) {
        conn.weight = oldhebbian(conn.weight, maxweight, pre_activation,
                                 post_activation, hebb_rate, pre_rate, post_rate);
      } else {
        conn.weight = hebbian(conn.weight, maxweight, pre_activation,
                              post_activation, hebb_rate, pre_rate, post_rate);
      }
    }
  }
}

// Reset network state (activations, counters)
void brain::reset_state() {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].value = 0.0f;
    neurons[i].last_activation = 0.0f;
    neurons[i].last_activation2 = 0.0f;
    neurons[i].activation_count = 0.0f;
    neurons[i].visited = false;
    neurons[i].clear_override();

    // Reset link added_weight (for future learning params)
    for (auto& conn : neurons[i].in_connections) {
      conn.added_weight = 0.0f;
    }
  }
}

// rtNEAT: Flush network - reset all activations
// Reference: network.cpp flush methods
void brain::flush() {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].value = 0.0f;
    neurons[i].last_activation = 0.0f;
    neurons[i].last_activation2 = 0.0f;
    neurons[i].activation_count = 0.0f;
    neurons[i].visited = false;
    // Note: Don't clear overrides - they may be intentionally set
  }
}

// rtNEAT: Recursive flushback from a specific neuron
// Reference: nnode.cpp:206-235
void brain::flushback(size_t neuron_idx) {
  if (neuron_idx >= neurons.size()) return;

  auto& n = neurons[neuron_idx];

  // Sensors should not flush back
  if (n.type == INPUT || n.type == BIAS) {
    n.activation_count = 0.0f;
    n.value = 0.0f;
    n.last_activation = 0.0f;
    n.last_activation2 = 0.0f;
    return;
  }

  if (n.activation_count > 0) {
    n.activation_count = 0.0f;
    n.value = 0.0f;
    n.last_activation = 0.0f;
    n.last_activation2 = 0.0f;

    // Flush back recursively through incoming connections
    for (auto& conn : n.in_connections) {
      conn.added_weight = 0.0f;
      if (neurons[conn.from_neuron].activation_count > 0) {
        flushback(conn.from_neuron);
      }
    }
  }
}

// Load input values using sensor_load (proper time-delay shifting)
void brain::load_sensors(const std::vector<exfloat> &input) {
  for (size_t i = 0; i < input.size() && i < input_neurons.size(); i++) {
    neurons[input_neurons[i]].sensor_load(input[i]);
  }
  // Also set bias neurons
  for (size_t i = 0; i < bias_neurons.size(); i++) {
    neurons[bias_neurons[i]].sensor_load(1.0f);
  }
}

// Override specific output neurons
void brain::override_outputs(const std::vector<exfloat> &values) {
  for (size_t i = 0; i < values.size() && i < output_neurons.size(); i++) {
    neurons[output_neurons[i]].override_output(values[i]);
  }
}

// Clear all overrides
void brain::clear_overrides() {
  for (auto& n : neurons) {
    n.clear_override();
  }
}

// Check if network has finished activating (all outputs have values)
bool brain::outputs_ready() const {
  for (size_t i = 0; i < output_neurons.size(); i++) {
    if (neurons[output_neurons[i]].activation_count == 0) {
      return false;
    }
  }
  return true;
}
