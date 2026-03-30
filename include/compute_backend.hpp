#ifndef ENEAT_COMPUTE_BACKEND_HPP
#define ENEAT_COMPUTE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "brain.hpp"

enum class backend_type { CPU, NPU, GPU };

// Abstract inference backend for production deployment
struct compute_backend {
  virtual ~compute_backend() = default;
  virtual void load(const brain &b) = 0;
  virtual void evaluate(const std::vector<exfloat> &input,
                        std::vector<exfloat> &output) = 0;
  virtual backend_type type() const = 0;
  virtual const char *name() const = 0;
};

// CPU backend - wraps existing brain evaluation
struct cpu_backend : compute_backend {
  brain net;

  void load(const brain &b) override {
    // Default memberwise copy (all members are copyable)
    net.neurons = b.neurons;
    net.traits = b.traits;
    net.recurrent = b.recurrent;
    net.hebbian_enabled = b.hebbian_enabled;
    net.input_neurons = b.input_neurons;
    net.bias_neurons = b.bias_neurons;
    net.output_neurons = b.output_neurons;
  }

  void evaluate(const std::vector<exfloat> &input,
                std::vector<exfloat> &output) override {
    net.evaluate(input, output);
  }

  backend_type type() const override { return backend_type::CPU; }
  const char *name() const override { return "CPU"; }
};

#ifdef ENEAT_NPU_ENABLED
// NPU backend using ESWIN ESSDK runtime
// Loads pre-compiled .model files produced by EsAAC
struct npu_backend : compute_backend {
  struct impl;
  std::unique_ptr<impl> pimpl;

  npu_backend();
  ~npu_backend();

  // Load from a compiled .model file (the NPU path)
  bool load_model(const std::string &model_path);

  // Not supported - NPU requires pre-compiled models
  // Use export_onnx() → EsQuant → EsAAC → .model → load_model()
  void load(const brain &b) override;

  void evaluate(const std::vector<exfloat> &input,
                std::vector<exfloat> &output) override;

  backend_type type() const override { return backend_type::NPU; }
  const char *name() const override { return "NPU (ESWIN EIC7700)"; }
};
#endif

#endif
