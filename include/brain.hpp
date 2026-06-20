#ifndef ENEAT_BRAIN_HPP
#define ENEAT_BRAIN_HPP

#include <fstream>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stack>
#include <stddef.h>
#include <streambuf>
#include <string>
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
  // Number of activation (synapse-hop) passes to run per evaluate() call so
  // that each item's outputs reflect the item actually presented rather than a
  // single lagged hop. Derived from network depth at operator=(genome).
  size_t settle_passes = 3;
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

  // Lamarckian Hebbian write-back: copy each connection's (possibly Hebbian-
  // learned) weight back into the matching ENABLED gene of g, identified by the
  // (from_node, to_node) pair recorded on the connection. Only gene.weight is
  // modified; innovation numbers, enabled flags, structure, traits and node ids
  // are left untouched. Disabled genes and connections with unknown node ids
  // (src_node/dst_node == SIZE_MAX) are skipped.
  void write_back_to(genome &g) const;

  // Compute the max feed-forward depth (longest non-recurrent path from a
  // sensor to an output) and store it in settle_passes so evaluate_recurrent
  // propagates enough hops to de-lag per-item scoring.
  void compute_settle_passes();

  // Debug: Get fingerprint of brain structure and weights
  std::string fingerprint() const;
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
