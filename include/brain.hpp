#ifndef ENEAT_BRAIN_HPP
#define ENEAT_BRAIN_HPP

#include <fstream>
#include <iterator>
#include <ostream>
#include <stack>
#include <stddef.h>
#include <streambuf>
#include <vector>

#include "functions.hpp"
#include "genome.hpp"
#include "math.hpp"
#include "neuron.hpp"
#include "trait.hpp"

struct brain {
  std::vector<neuron> neurons;
  std::vector<eneat::trait> traits;  // Traits for Hebbian learning
  bool recurrent = false;
  bool hebbian_enabled = false;      // Enable Hebbian weight modification
  std::vector<size_t> input_neurons;
  std::vector<size_t> bias_neurons;
  std::vector<size_t> output_neurons;

  void operator=(const genome &g);
  inline void evaluate(const std::vector<exfloat> &input,
                       std::vector<exfloat> &output);
  void evaluate_nonrecurrent(const std::vector<exfloat> &input,
                             std::vector<exfloat> &output);
  void evaluate_recurrent(const std::vector<exfloat> &input,
                          std::vector<exfloat> &output);

  // Hebbian learning: apply weight modifications based on activations
  void apply_hebbian_learning();

  // rtNEAT Hebbian learning algorithms
  // oldhebbian: Original rtNEAT algorithm
  exfloat oldhebbian(exfloat weight, exfloat maxweight, exfloat active_in,
                     exfloat active_out, exfloat hebb_rate,
                     exfloat pre_rate, exfloat post_rate);

  // hebbian: Floreano & Urzelai 2000 algorithm
  exfloat hebbian(exfloat weight, exfloat maxweight, exfloat active_in,
                  exfloat active_out, exfloat hebb_rate,
                  exfloat pre_rate, exfloat post_rate);

  // Reset network state (activations, counters)
  void reset_state();

  // rtNEAT: Flush network - reset all activations
  // Reference: network.cpp flush methods
  void flush();

  // rtNEAT: Recursive flushback from a specific neuron
  // Reference: nnode.cpp:206-235
  void flushback(size_t neuron_idx);

  // Load input values using sensor_load (proper time-delay shifting)
  void load_sensors(const std::vector<exfloat> &input);

  // Override specific output neurons
  void override_outputs(const std::vector<exfloat> &values);

  // Clear all overrides
  void clear_overrides();

  // Check if network has finished activating (all outputs have values)
  bool outputs_ready() const;
};

std::istream &operator>>(std::istream &input, brain &b);
std::ostream &operator<<(std::ostream &output, brain &b);

void brain::evaluate(const std::vector<exfloat> &input,
                     std::vector<exfloat> &output) {
  if (recurrent)
    this->evaluate_recurrent(input, output);
  else
    this->evaluate_nonrecurrent(input, output);
}

#endif
