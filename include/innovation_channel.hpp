#ifndef ENEAT_INNOVATION_CHANNEL_HPP
#define ENEAT_INNOVATION_CHANNEL_HPP

#include <functional>
#include <memory>
#include <utility>

#include "coro_task.hpp"
#include "gene.hpp"
#include "innovation_container.hpp"
#include "shared_state.hpp"
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
  innovation_channel() = default;

  // Send a request and get the innovation number via callback
  void request_innovation(size_t from_node, size_t to_node,
                          std::function<void(size_t)> callback) {
    requests_.send({from_node, to_node, std::move(callback)});
  }

  // Synchronous request (blocks until response) - requires service running
  size_t request_innovation_sync(size_t from_node, size_t to_node) {
    ethreads::auto_reset_event done;
    size_t result{0};

    request_innovation(from_node, to_node, [&](size_t inn_num) {
      result = inn_num;
      done.set();
    });

    // Block until service processes request
    done.wait();

    return result;
  }

  // Coroutine-friendly request - yields while waiting
  ethreads::coro_task<size_t> request_innovation_async(size_t from_node,
                                                        size_t to_node) {
    ethreads::async_auto_event done;
    size_t result{0};

    request_innovation(from_node, to_node, [&](size_t inn_num) {
      result = inn_num;
      done.set();
    });

    // Await until service processes request
    co_await done;

    co_return result;
  }

  // Process pending requests (call from service coroutine)
  void process_one() {
    auto req = requests_.try_receive();
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
    running_.set();  // Allow service to run again
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
  void stop() { running_.reset(); }

  bool is_running() const { return running_.is_set(); }

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
  ethreads::channel<innovation_request> requests_;
  ethreads::manual_reset_event running_{true};
};

} // namespace eneat

#endif
