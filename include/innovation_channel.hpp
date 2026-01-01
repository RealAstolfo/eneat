#ifndef ENEAT_INNOVATION_CHANNEL_HPP
#define ENEAT_INNOVATION_CHANNEL_HPP

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <utility>

#include "coro_task.hpp"
#include "gene.hpp"
#include "innovation_container.hpp"
#include "ts_queue.hpp"
#include "yield.hpp"

namespace eneat {

// Request to add a gene and get its innovation number
struct innovation_request {
  size_t from_node;
  size_t to_node;
  std::function<void(size_t)> callback;
};

// Channel-based innovation service
// Single owner processes all requests, eliminating lock contention
class innovation_channel {
public:
  innovation_channel() : running_(true) {}

  // Send a request and get the innovation number via callback
  void request_innovation(size_t from_node, size_t to_node,
                          std::function<void(size_t)> callback) {
    requests_.push({from_node, to_node, std::move(callback)});
  }

  // Synchronous request (blocks until response) - requires service running
  size_t request_innovation_sync(size_t from_node, size_t to_node) {
    std::atomic<bool> done{false};
    std::atomic<size_t> result{0};

    request_innovation(from_node, to_node, [&](size_t inn_num) {
      result.store(inn_num, std::memory_order_relaxed);
      done.store(true, std::memory_order_release);
    });

    // Spin-wait for result (service must be running)
    while (!done.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    return result.load(std::memory_order_relaxed);
  }

  // Coroutine-friendly request - yields while waiting
  ethreads::coro_task<size_t> request_innovation_async(size_t from_node,
                                                        size_t to_node) {
    std::atomic<bool> done{false};
    std::atomic<size_t> result{0};

    request_innovation(from_node, to_node, [&](size_t inn_num) {
      result.store(inn_num, std::memory_order_relaxed);
      done.store(true, std::memory_order_release);
    });

    // Yield while waiting for service to process
    while (!done.load(std::memory_order_acquire)) {
      co_await ethreads::yield();
    }

    co_return result.load(std::memory_order_relaxed);
  }

  // Process pending requests (call from service coroutine)
  void process_one() {
    auto req = requests_.pop();
    if (req) {
      gene g;
      g.from_node = req->from_node;
      g.to_node = req->to_node;
      size_t inn_num = container_.add_gene(g);
      req->callback(inn_num);
    }
  }

  // Process all pending requests
  void process_all() {
    while (!requests_.empty()) {
      process_one();
    }
  }

  // Reset innovation tracking (called at generation start)
  void reset() {
    container_.reset();
    running_.store(true, std::memory_order_release);  // Allow service to run again
  }

  // Set innovation number (for loading saved state)
  void set_innovation_number(size_t num) {
    container_.set_innovation_number(num);
  }

  // Get current innovation number
  size_t number() { return container_.number(); }

  // Direct synchronous access (for single-threaded init only)
  // Use request_innovation_sync/async during concurrent operations
  size_t add_gene_direct(size_t from_node, size_t to_node) {
    gene g;
    g.from_node = from_node;
    g.to_node = to_node;
    return container_.add_gene(g);
  }

  // Stop the service
  void stop() { running_.store(false, std::memory_order_release); }

  bool is_running() const { return running_.load(std::memory_order_acquire); }

  // Service coroutine - runs continuously processing requests
  ethreads::coro_task<void> run_service() {
    while (is_running()) {
      process_all();
      co_await ethreads::yield();
    }
    co_return;
  }

private:
  innovation_container container_;
  ts_queue<innovation_request> requests_;
  std::atomic<bool> running_;
};

} // namespace eneat

#endif
