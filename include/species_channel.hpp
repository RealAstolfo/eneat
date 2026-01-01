#ifndef ENEAT_SPECIES_CHANNEL_HPP
#define ENEAT_SPECIES_CHANNEL_HPP

#include <functional>
#include <list>
#include <memory>
#include <utility>

#include "coro_task.hpp"
#include "genome.hpp"
#include "specie.hpp"
#include "shared_state.hpp"
#include "yield.hpp"
#include "async_runtime.hpp"

namespace eneat {

// Request to add a genome to species
struct species_request {
  genome child;
  std::function<void()> callback;
};

// Channel-based species management service
// Single owner processes all add_to_species requests, eliminating race conditions
class species_channel {
public:
  species_channel(std::list<specie> &species_list,
                  std::function<bool(const genome &, const genome &)> same_species_fn)
      : species_(species_list), is_same_species_(same_species_fn) {}

  // Send a request to add a genome to species
  void request_add(genome child, std::function<void()> callback = nullptr) {
    requests_.send({std::move(child), std::move(callback)});
  }

  // Synchronous add (blocks until complete) - requires service running
  void request_add_sync(genome child) {
    ethreads::auto_reset_event done;

    request_add(std::move(child), [&]() {
      done.set();
    });

    // Block until service processes request
    done.wait();
  }

  // Coroutine-friendly add - yields while waiting
  ethreads::coro_task<void> request_add_async(genome child) {
    ethreads::async_auto_event done;

    request_add(std::move(child), [&]() {
      done.set();
    });

    // Await until service processes request
    co_await done;

    co_return;
  }

  // Process one pending request
  void process_one() {
    auto req = requests_.try_receive();
    if (req) {
      add_to_species_impl(std::move(req->child));
      if (req->callback) {
        req->callback();
      }
    }
  }

  // Process all pending requests
  void process_all() {
    while (!requests_.empty()) {
      process_one();
    }
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

  // Get pending request count
  bool has_pending() { return !requests_.empty(); }

private:
  // Internal implementation of add_to_species
  void add_to_species_impl(genome child) {
    auto s = species_.begin();
    while (s != species_.end()) {
      if (is_same_species_(child, s->genomes[0])) {
        s->genomes.push_back(std::move(child));
        return;
      }
      s++;
    }

    // No matching species found, create new one
    specie new_specie;
    new_specie.genomes.push_back(std::move(child));
    species_.push_back(std::move(new_specie));
  }

  std::list<specie> &species_;
  std::function<bool(const genome &, const genome &)> is_same_species_;
  ethreads::channel<species_request> requests_;
  ethreads::manual_reset_event running_{true};
};

} // namespace eneat

#endif
