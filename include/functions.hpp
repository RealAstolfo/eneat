#ifndef ENEAT_FUNCTIONS_HPP
#define ENEAT_FUNCTIONS_HPP

#include <cmath>
#include <iostream>

#include "math.hpp"

#define M_PI 3.14159265358979323846 /* pi */

typedef enum {
  FIRST = 0,
  RELU = 0,
  LINEAR = 1,
  HEAVISIDE = 2,
  LOGISTIC = 3,
  SIGMOID = 4,
  TANH = 5,
  GELU = 6,
  SWISH = 7,
  LEAKY_RELU = 8,
  NORMALIZE = 9,
  LAST = 9,
} ai_func_type;

// basic, and most relied on activation function
inline exfloat relu(exfloat x) { return (x > 0) ? x : 0; }

inline exfloat linear(exfloat x) { return x; }

inline exfloat heaviside(exfloat x) { return (x >= 0) ? 1 : 0; }

inline exfloat logistic(exfloat x) noexcept { return 1 / (1 + std::exp(-x)); }

inline exfloat sigmoid(exfloat x) noexcept {
  return 2 / (1 + std::exp(-x)) - 1;
}

using std::tanh;

inline exfloat gelu(exfloat x) {
  return 0.5f * x *
         (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * pow(x, 3))));
}

inline exfloat swish(exfloat x) { return x * sigmoid(x); }

inline exfloat leaky_relu(exfloat x, exfloat alpha = 0.01f) {
  return (x > 0) ? x : alpha * x;
}

inline exfloat normalize(exfloat x, exfloat alpha = 1.0f) {
  return (alpha != 0.0f) ? x / alpha : 1.0f;
}

#endif
