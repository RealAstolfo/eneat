#ifndef ENEAT_TRAIT_HPP
#define ENEAT_TRAIT_HPP

#include "math.hpp"
#include <array>
#include <iostream>
#include <random>

namespace eneat {

// Number of trait parameters (matches rtNEAT)
constexpr size_t NUM_TRAIT_PARAMS = 8;

// Trait parameter indices for Hebbian learning
enum trait_param : size_t {
  HEBBIAN_RATE = 0,      // Weight change from pre * post correlation
  PRESYNAPTIC_RATE = 1,  // Weight change from presynaptic activity
  POSTSYNAPTIC_RATE = 2, // Weight change from postsynaptic activity
  MAX_WEIGHT = 3,        // Maximum weight bound
  MIN_WEIGHT = 4,        // Minimum weight bound
  LEARNING_SIGNAL = 5,   // Modulation signal for learning
  RESERVED_1 = 6,
  RESERVED_2 = 7
};

// A Trait is a group of parameters that can be expressed as a group
// more than one time. Traits save a genetic algorithm from having to
// search vast parameter landscapes on every node. Instead, each node
// can simply point to a trait and those traits can evolve on their own.
struct trait {
  size_t trait_id = 0;
  std::array<exfloat, NUM_TRAIT_PARAMS> params{};

  trait() = default;

  trait(size_t id) : trait_id(id) {
    params.fill(0.0f);
  }

  trait(size_t id, exfloat p0, exfloat p1, exfloat p2, exfloat p3,
        exfloat p4, exfloat p5, exfloat p6, exfloat p7)
      : trait_id(id), params{p0, p1, p2, p3, p4, p5, p6, p7} {}

  // Create a trait that is the average of two traits
  static trait average(const trait &t1, const trait &t2) {
    trait result;
    result.trait_id = t1.trait_id; // Use first trait's ID
    for (size_t i = 0; i < NUM_TRAIT_PARAMS; i++) {
      result.params[i] = (t1.params[i] + t2.params[i]) / 2.0f;
    }
    return result;
  }

  // Perturb the trait parameters slightly
  void mutate(exfloat mutation_power = 0.1f, exfloat mutation_prob = 0.5f) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<exfloat> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<exfloat> mut_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < NUM_TRAIT_PARAMS; i++) {
      if (prob_dist(rng) < mutation_prob) {
        params[i] += mut_dist(rng) * mutation_power;
        // Clamp to [0, 1] range
        if (params[i] < 0.0f)
          params[i] = 0.0f;
        if (params[i] > 1.0f)
          params[i] = 1.0f;
      }
    }
  }

  // Check if this trait enables Hebbian learning
  // Traits with ID 2, 3, or 4 are learning-enabled (rtNEAT convention)
  bool is_learning_enabled() const {
    return trait_id >= 2 && trait_id <= 4;
  }

  // Get Hebbian learning parameters
  exfloat hebbian_rate() const { return params[HEBBIAN_RATE]; }
  exfloat presynaptic_rate() const { return params[PRESYNAPTIC_RATE]; }
  exfloat postsynaptic_rate() const { return params[POSTSYNAPTIC_RATE]; }
  exfloat max_weight() const { return params[MAX_WEIGHT]; }
  exfloat min_weight() const { return params[MIN_WEIGHT]; }
};

// Serialization operators
inline std::ostream& operator<<(std::ostream& out, const trait& t) {
  out << t.trait_id;
  for (size_t i = 0; i < NUM_TRAIT_PARAMS; i++) {
    out << " " << t.params[i];
  }
  return out;
}

inline std::istream& operator>>(std::istream& in, trait& t) {
  in >> t.trait_id;
  for (size_t i = 0; i < NUM_TRAIT_PARAMS; i++) {
    in >> t.params[i];
  }
  return in;
}

} // namespace eneat

#endif
