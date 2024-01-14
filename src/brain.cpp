#include "brain.hpp"
#include "functions.hpp"
#include "gene.hpp"
#include "neuron.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <ostream>
#include <stddef.h>
#include <unordered_map>
#include <utility>
#include <vector>

std::istream &operator>>(std::istream &input, brain &b) {
  b.neurons.clear();
  b.input_neurons.clear();
  b.output_neurons.clear();
  input >> b.recurrent;
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
    b.neurons[i].visited = false;
    input >> tmp_type;
    type = (neuron_place)tmp_type;
    input >> tmp_act;
    act = (ai_func_type)tmp_act;
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
      size_t t;
      exfloat w;
      input >> t >> w;
      b.neurons[i].in_neurons.push_back(std::make_pair(t, w));
    }
  }
  return input;
}

std::ostream &operator<<(std::ostream &output, brain &b) {
  output << b.recurrent << std::endl;
  output << b.neurons.size() << std::endl << std::endl;
  for (size_t i = 0; i < b.neurons.size(); i++) {
    output << (size_t)b.neurons[i].type << " ";
    output << (size_t)b.neurons[i].activation_function << " ";
    output << b.neurons[i].in_neurons.size() << std::endl;
    for (size_t j = 0; j < b.neurons[i].in_neurons.size(); j++)
      output << b.neurons[i].in_neurons[j].first << " "
             << b.neurons[i].in_neurons[j].second << " ";
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
    if (table.find(it->second.from_node) == table.end()) {
      neurons.push_back(n);
      table[it->second.from_node] = neurons.size() - 1;
    }
    if (table.find(it->second.to_node) == table.end()) {
      neurons.push_back(n);
      table[it->second.to_node] = neurons.size() - 1;
    }
  }

  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    neurons[table[it->second.to_node]].in_neurons.push_back(
        std::make_pair(table[it->second.from_node], it->second.weight));
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
      for (size_t i = 0; i < neurons[t].in_neurons.size(); i++)
        sum += neurons[neurons[t].in_neurons[i].first].value *
               neurons[t].in_neurons[i].second;
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
      for (size_t i = 0; i < neurons[t].in_neurons.size(); i++)
        if (!neurons[neurons[t].in_neurons[i].first].visited)
          s.push(neurons[t].in_neurons[i].first);
    }
  }

  for (size_t i = 0; i < output_neurons.size() && i < output.size(); i++)
    output[i] = neurons[output_neurons[i]].value;
}

void brain::evaluate_recurrent(const std::vector<exfloat> &input,
                               std::vector<exfloat> &output) {
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
    for (size_t j = 0; j < neurons[i].in_neurons.size(); j++)
      sum += neurons[neurons[i].in_neurons[j].first].value *
             neurons[i].in_neurons[j].second;
    if (neurons[i].in_neurons.size() > 0) {
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
  }

  for (size_t i = 0; i < output_neurons.size() && i < output.size(); i++)
    output[i] = neurons[output_neurons[i]].value;
}
