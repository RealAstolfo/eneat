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
      input >> from >> w >> tid >> rec;
      b.neurons[i].in_connections.emplace_back(from, w, tid, rec);
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
             << conn.trait_id << " " << conn.is_recurrent << " ";
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

  // Add connections with trait information
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (!it->second.enabled)
      continue;
    neurons[table[it->second.to_node]].in_connections.emplace_back(
        table[it->second.from_node],
        it->second.weight,
        it->second.trait_id,
        it->second.is_recurrent
    );
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
  // Store last activations for Hebbian learning
  for (size_t i = 0; i < neurons.size(); i++) {
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
    for (size_t j = 0; j < neurons[i].in_connections.size(); j++)
      sum += neurons[neurons[i].in_connections[j].from_neuron].value *
             neurons[i].in_connections[j].weight;
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

// Hebbian learning: modify weights based on pre/post synaptic activations
// Formula: delta_w = A*pre*post + B*pre + C*post + D
// Where A,B,C,D come from the trait parameters
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

      // Hebbian learning rule:
      // params[0] = Hebbian coefficient (pre * post)
      // params[1] = Presynaptic coefficient (pre only)
      // params[2] = Postsynaptic coefficient (post only)
      // params[3] = Bias/decay term
      // params[4] = Learning rate
      // params[5] = Weight limit (max absolute value)

      exfloat delta_w = t.params[0] * pre_activation * post_activation +
                        t.params[1] * pre_activation +
                        t.params[2] * post_activation +
                        t.params[3];

      // Apply learning rate
      delta_w *= t.params[4];

      // Update weight
      conn.weight += delta_w;

      // Clamp weight to limit
      exfloat limit = std::abs(t.params[5]);
      if (limit > 0.0f) {
        if (conn.weight > limit) conn.weight = limit;
        if (conn.weight < -limit) conn.weight = -limit;
      }
    }
  }
}

// Reset network state (activations, counters)
void brain::reset_state() {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].value = 0.0f;
    neurons[i].last_activation = 0.0f;
    neurons[i].activation_count = 0.0f;
    neurons[i].visited = false;
  }
}
