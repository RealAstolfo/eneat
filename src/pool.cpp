#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <stddef.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "async_runtime.hpp"
#include "coro_task.hpp"
#include "functions.hpp"
#include "gene.hpp"
#include "genome.hpp"
#include "network_info_container.hpp"
#include "neuron.hpp"
#include "pool.hpp"
#include "specie.hpp"
#include <task_scheduler.hpp>

pool::pool(size_t input, size_t output, size_t bias, bool rec) {
  network_info.input_size = input;
  network_info.output_size = output;
  network_info.bias_size = bias;
  network_info.functional_neurons = input + output + bias;
  network_info.recurrent = rec;
  // Thread-local RNG is initialized per-thread, no seeding needed here
  for (size_t i = 0; i < speciating_parameters.population; i++) {
    genome new_genome(network_info, mutation_rates);
    mutate_sync(new_genome);  // Use sync version for initialization
    add_to_species(new_genome);
  }
}

std::vector<std::pair<specie *, genome *>> pool::get_genomes() {
  std::vector<std::pair<specie *, genome *>> result;
  for (auto s = this->species.begin(); s != this->species.end(); s++) {
    s->genomes.modify([&](std::vector<genome> &g) {
      for (size_t i = 0; i < g.size(); i++)
        result.push_back(std::make_pair(&(*s), &g[i]));
    });
  }
  return result;
}

// Default crossover - delegates to multipoint
genome pool::crossover(const genome &g1, const genome &g2) {
  return crossover_multipoint(g1, g2);
}

// Multipoint crossover: randomly select from matching genes, take excess from fitter parent
genome pool::crossover_multipoint(const genome &g1, const genome &g2) {
  // Ensure g1 is the fitter parent (or smaller if equal fitness)
  if (g2.fitness.load() > g1.fitness.load())
    return crossover_multipoint(g2, g1);
  if (g2.fitness.load() == g1.fitness.load() && g2.genes.size() < g1.genes.size())
    return crossover_multipoint(g2, g1);

  genome child(network_info, mutation_rates);
  // Average traits from both parents
  for (size_t i = 0; i < std::min(g1.traits.size(), g2.traits.size()); i++) {
    child.traits.push_back(eneat::trait::average(g1.traits[i], g2.traits[i]));
  }
  // Copy remaining traits from fitter parent
  for (size_t i = child.traits.size(); i < g1.traits.size(); i++) {
    child.traits.push_back(g1.traits[i]);
  }

  std::uniform_int_distribution<int> coin_flip(1, 2);
  std::uniform_real_distribution<exfloat> prob_dist(0.0f, 1.0f);

  for (auto it1 = g1.genes.begin(); it1 != g1.genes.end(); it1++) {
    auto it2 = g2.genes.find(it1->second.innovation_num);
    gene child_gene;

    if (it2 != g2.genes.end()) {
      // Matching gene - randomly select from either parent
      int coin = eneat::get_rand(coin_flip);
      if (coin == 2)
        child_gene = it2->second;
      else
        child_gene = it1->second;

      // 75% disabled rule: if either parent has gene disabled, 75% chance child inherits disabled
      if (!it1->second.enabled || !it2->second.enabled) {
        if (eneat::get_rand(prob_dist) < 0.75f) {
          child_gene.enabled = false;
        }
      }
    } else {
      // Disjoint/excess gene - take from fitter parent (g1)
      child_gene = it1->second;
    }

    child.genes[child_gene.innovation_num] = child_gene;
  }

  child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
  child.can_be_recurrent = g1.can_be_recurrent || g2.can_be_recurrent;
  return child;
}

// Multipoint average crossover: average weights of matching genes
genome pool::crossover_multipoint_avg(const genome &g1, const genome &g2) {
  // Ensure g1 is the fitter parent
  if (g2.fitness.load() > g1.fitness.load())
    return crossover_multipoint_avg(g2, g1);
  if (g2.fitness.load() == g1.fitness.load() && g2.genes.size() < g1.genes.size())
    return crossover_multipoint_avg(g2, g1);

  genome child(network_info, mutation_rates);
  // Average traits
  for (size_t i = 0; i < std::min(g1.traits.size(), g2.traits.size()); i++) {
    child.traits.push_back(eneat::trait::average(g1.traits[i], g2.traits[i]));
  }
  for (size_t i = child.traits.size(); i < g1.traits.size(); i++) {
    child.traits.push_back(g1.traits[i]);
  }

  std::uniform_real_distribution<exfloat> prob_dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> coin_flip(1, 2);

  for (auto it1 = g1.genes.begin(); it1 != g1.genes.end(); it1++) {
    auto it2 = g2.genes.find(it1->second.innovation_num);
    gene child_gene;

    if (it2 != g2.genes.end()) {
      // Matching gene - average weights
      child_gene = it1->second;
      child_gene.weight = (it1->second.weight + it2->second.weight) / 2.0f;
      child_gene.mutation_num = (it1->second.mutation_num + it2->second.mutation_num) / 2.0f;

      // Randomly pick activation function
      if (eneat::get_rand(coin_flip) == 2) {
        child_gene.activation = it2->second.activation;
      }

      // 75% disabled rule
      if (!it1->second.enabled || !it2->second.enabled) {
        if (eneat::get_rand(prob_dist) < 0.75f) {
          child_gene.enabled = false;
        }
      }
    } else {
      child_gene = it1->second;
    }

    child.genes[child_gene.innovation_num] = child_gene;
  }

  child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
  child.can_be_recurrent = g1.can_be_recurrent || g2.can_be_recurrent;
  return child;
}

// Single-point crossover: select crossover point, take genes from each parent
genome pool::crossover_singlepoint(const genome &g1, const genome &g2) {
  genome child(network_info, mutation_rates);

  // Average traits
  for (size_t i = 0; i < std::min(g1.traits.size(), g2.traits.size()); i++) {
    child.traits.push_back(eneat::trait::average(g1.traits[i], g2.traits[i]));
  }

  // Find crossover point in the smaller genome
  size_t smaller_size = std::min(g1.genes.size(), g2.genes.size());
  if (smaller_size == 0) {
    child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
    return child;
  }

  std::uniform_int_distribution<size_t> point_dist(0, smaller_size - 1);
  size_t crossover_point = eneat::get_rand(point_dist);

  std::uniform_real_distribution<exfloat> prob_dist(0.0f, 1.0f);

  // Get iterators
  auto it1 = g1.genes.begin();
  auto it2 = g2.genes.begin();

  // Take genes before crossover point from g1, after from g2
  size_t index = 0;
  while (it1 != g1.genes.end() || it2 != g2.genes.end()) {
    gene child_gene;
    bool use_g1 = (index < crossover_point);

    if (it1 != g1.genes.end() && it2 != g2.genes.end()) {
      if (it1->second.innovation_num == it2->second.innovation_num) {
        // Matching gene
        child_gene = use_g1 ? it1->second : it2->second;
        // 75% disabled rule
        if (!it1->second.enabled || !it2->second.enabled) {
          if (eneat::get_rand(prob_dist) < 0.75f) {
            child_gene.enabled = false;
          }
        }
        ++it1;
        ++it2;
      } else if (it1->second.innovation_num < it2->second.innovation_num) {
        child_gene = it1->second;
        ++it1;
      } else {
        child_gene = it2->second;
        ++it2;
      }
    } else if (it1 != g1.genes.end()) {
      child_gene = it1->second;
      ++it1;
    } else {
      child_gene = it2->second;
      ++it2;
    }

    child.genes[child_gene.innovation_num] = child_gene;
    index++;
  }

  child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
  child.can_be_recurrent = g1.can_be_recurrent || g2.can_be_recurrent;
  return child;
}

ethreads::coro_task<void> pool::mutate_activation(genome &g) {
  std::uniform_int_distribution<size_t> distributor(ai_func_type::FIRST,
                                                    ai_func_type::LAST);
  std::uniform_real_distribution<exfloat> coin(0.0f, 1.0f);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (eneat::get_rand(coin) < g.mutation_rates.activation_mutation_chance)
      it->second.activation = (ai_func_type)eneat::get_rand(distributor);
  }
  co_return;
}

ethreads::coro_task<void> pool::mutate_weight(genome &g) {
  exfloat step = mutation_rates.step_size;
  std::uniform_real_distribution<exfloat> real_distributor(0.0f, 1.0f);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    // Skip frozen genes (cannot be mutated)
    if (it->second.frozen)
      continue;

    if (eneat::get_rand(real_distributor) < mutation_rates.perturb_chance) {
      it->second.weight +=
          eneat::get_rand(real_distributor) * step * 2.0f - step;
    } else {
      it->second.weight = eneat::get_rand(real_distributor) * 4.0f - 2.0f;
    }

    // Track mutation for compatibility distance calculation
    it->second.mutation_num += 1.0f;
  }
  co_return;
}

ethreads::coro_task<void> pool::mutate_enable_disable(genome &g, const bool &enable) {
  std::vector<gene *> v;
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    if (it->second.enabled != enable)
      v.push_back(&it->second);
  if (v.size() == 0)
    co_return;
  std::uniform_int_distribution<int> distributor(0, v.size() - 1);
  v[eneat::get_rand(distributor)]->enabled = enable;
  co_return;
}

inline bool is_input(size_t &neuron, network_info_container &network_info) {
  return neuron < network_info.input_size;
}

inline bool is_output(size_t &neuron, network_info_container &network_info) {
  return neuron < network_info.functional_neurons &&
         neuron >= (network_info.input_size + network_info.bias_size);
}

inline bool is_bias(size_t &neuron, network_info_container &network_info) {
  return neuron < (network_info.input_size + network_info.bias_size) &&
         neuron >= network_info.input_size;
}

ethreads::coro_task<void> pool::mutate_link(genome &g, const bool &force_bias) {
  std::uniform_int_distribution<size_t> distributor1(0, g.max_neuron - 1);
  size_t neuron1 = eneat::get_rand(distributor1);
  std::uniform_int_distribution<size_t> distributor2(
      network_info.input_size + network_info.bias_size, g.max_neuron - 1);
  size_t neuron2 = eneat::get_rand(distributor2);
  if (is_output(neuron1, network_info) && is_output(neuron2, network_info))
    co_return;
  if (is_bias(neuron2, network_info))
    co_return;
  if (neuron1 == neuron2 && (!force_bias))
    co_return;
  if (is_output(neuron1, network_info))
    std::swap(neuron1, neuron2);
  if (force_bias) {
    std::uniform_int_distribution<size_t> bias_choose(
        network_info.input_size,
        network_info.input_size + network_info.output_size - 1);
    neuron1 = eneat::get_rand(bias_choose);
  }
  if (!g.network_info.recurrent) {
    bool has_recurrence = false;
    // Bias and input nodes cannot create cycles when used as source
    if (is_bias(neuron1, network_info) || is_input(neuron1, network_info))
      has_recurrence = false;
    else {
      // Build adjacency list of current connections
      std::unordered_map<size_t, std::vector<size_t>> connections;
      for (const auto &gene : g.genes) {
        if (gene.second.enabled) {
          connections[gene.second.from_node].push_back(gene.second.to_node);
        }
      }
      // Add the proposed new connection
      connections[neuron1].push_back(neuron2);

      // BFS from neuron2 (destination) to check if we can reach neuron1 (source)
      // If we can, adding neuron1->neuron2 would create a cycle
      std::queue<size_t> que;
      que.push(neuron2);  // Start from destination
      std::unordered_set<size_t> visited;
      visited.insert(neuron2);
      has_recurrence = false;

      while (!que.empty()) {
        size_t tmp = que.front();
        que.pop();

        // If we reach the source from the destination, it's a cycle
        if (tmp == neuron1) {
          has_recurrence = true;
          break;
        }

        for (const auto &next_node : connections[tmp]) {
          if (visited.find(next_node) == visited.end()) {
            que.push(next_node);
            visited.insert(next_node);
          }
        }
      }
    }

    if (has_recurrence)
      co_return;
  }

  gene new_gene;
  new_gene.from_node = neuron1;
  new_gene.to_node = neuron2;
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    if (it->second.from_node == neuron1 && it->second.to_node == neuron2)
      co_return;

  // Use channel-based innovation tracking (async)
  new_gene.innovation_num = co_await innovation_chan.request_innovation_async(
      new_gene.from_node, new_gene.to_node);
  std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> act_generator(ai_func_type::FIRST,
                                                      ai_func_type::LAST);
  new_gene.weight = eneat::get_rand(weight_generator) * 4.0f - 2.0f;
  new_gene.activation = (ai_func_type)eneat::get_rand(act_generator);
  g.genes[new_gene.innovation_num] = new_gene;
  co_return;
}

ethreads::coro_task<void> pool::mutate_neuron(genome &g) {
  if (g.genes.size() == 0)
    co_return;
  g.max_neuron++;
  std::uniform_int_distribution<size_t> distributor(0, g.genes.size() - 1);
  size_t gene_id = eneat::get_rand(distributor);
  auto it = g.genes.begin();
  std::advance(it, gene_id);
  if (it->second.enabled == false)
    co_return;
  it->second.enabled = false;
  gene new_gene1;
  new_gene1.from_node = it->second.from_node;
  new_gene1.to_node = g.max_neuron - 1;
  // Per NEAT paper: first link (input -> new node) has weight 1.0
  // This preserves network behavior initially
  new_gene1.weight = 1.0f;
  std::uniform_int_distribution<size_t> act_dist(ai_func_type::FIRST,
                                                 ai_func_type::LAST);
  new_gene1.activation = (ai_func_type)eneat::get_rand(act_dist);
  // Use channel-based innovation tracking (async)
  new_gene1.innovation_num = co_await innovation_chan.request_innovation_async(
      new_gene1.from_node, new_gene1.to_node);
  new_gene1.enabled = true;
  gene new_gene2;
  new_gene2.from_node = g.max_neuron - 1;
  new_gene2.to_node = it->second.to_node;
  new_gene2.weight = it->second.weight;
  new_gene2.activation = it->second.activation;
  // Use channel-based innovation tracking (async)
  new_gene2.innovation_num = co_await innovation_chan.request_innovation_async(
      new_gene2.from_node, new_gene2.to_node);
  new_gene2.enabled = true;
  g.genes[new_gene1.innovation_num] = new_gene1;
  g.genes[new_gene2.innovation_num] = new_gene2;
  co_return;
}

ethreads::coro_task<void> pool::mutate_bias_neuron(genome &g) {
  // Create a new bias neuron
  size_t new_neuron_id = g.max_neuron++;

  // Build set of existing bias neuron IDs from genes
  std::unordered_set<size_t> bias_nodes;
  for (const auto &[_, gene] : g.genes) {
    if (gene.is_bias_source) {
      bias_nodes.insert(gene.from_node);
    }
  }

  // Create a connection from this bias neuron to a random hidden/output neuron
  std::uniform_int_distribution<size_t> distributor(
      network_info.input_size + network_info.bias_size, g.max_neuron - 2);
  size_t target = eneat::get_rand(distributor);

  // Skip if target is also an evolved bias neuron
  if (bias_nodes.count(target)) {
    co_return;  // Don't connect bias to bias
  }

  gene new_gene;
  new_gene.from_node = new_neuron_id;
  new_gene.to_node = target;
  new_gene.is_bias_source = true;  // Mark this gene as originating from a bias neuron
  new_gene.innovation_num = co_await innovation_chan.request_innovation_async(
      new_gene.from_node, new_gene.to_node);
  std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> act_generator(ai_func_type::FIRST,
                                                      ai_func_type::LAST);
  new_gene.weight = eneat::get_rand(weight_generator) * 4.0f - 2.0f;
  new_gene.activation = (ai_func_type)eneat::get_rand(act_generator);
  g.genes[new_gene.innovation_num] = new_gene;
  co_return;
}

ethreads::coro_task<void> pool::mutate(genome &g) {
  exfloat coefficient[2] = {0.95f, 1.05263f};
  std::uniform_int_distribution<int> coin_flip(0, 1);
  g.mutation_rates.enable_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.disable_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.connection_mutate_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.neuron_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.link_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.bias_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.bias_neuron_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.crossover_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.perturb_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.activation_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  std::uniform_real_distribution<exfloat> mutate_or_not_mutate(0.0f, 1.0f);
  if (eneat::get_rand(mutate_or_not_mutate) <
      g.mutation_rates.connection_mutate_chance)
    co_await mutate_weight(g);

  if (eneat::get_rand(mutate_or_not_mutate) <
      g.mutation_rates.activation_mutation_chance)
    co_await mutate_activation(g);

  exfloat p = g.mutation_rates.link_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_link(g, false);
    p -= 1.0f;
  }

  p = g.mutation_rates.neuron_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_neuron(g);
    p -= 1.0f;
  }

  p = g.mutation_rates.bias_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_link(g, true);
    p -= 1.0f;
  }

  p = g.mutation_rates.bias_neuron_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_bias_neuron(g);
    p -= 1.0f;
  }

  p = g.mutation_rates.enable_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_enable_disable(g, true);
    p -= 1.0f;
  }

  p = g.mutation_rates.disable_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p)
      co_await mutate_enable_disable(g, false);
    p -= 1.0f;
  }
  co_return;
}

// Synchronous mutation for single-threaded initialization
void pool::mutate_sync(genome &g) {
  exfloat coefficient[2] = {0.95f, 1.05263f};
  std::uniform_int_distribution<int> coin_flip(0, 1);
  g.mutation_rates.enable_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.disable_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.connection_mutate_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.neuron_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.link_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.bias_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.bias_neuron_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.crossover_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.perturb_chance *=
      coefficient[eneat::get_rand(coin_flip)];
  g.mutation_rates.activation_mutation_chance *=
      coefficient[eneat::get_rand(coin_flip)];

  std::uniform_real_distribution<exfloat> mutate_or_not_mutate(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> act_dist(ai_func_type::FIRST,
                                                 ai_func_type::LAST);
  std::uniform_real_distribution<exfloat> weight_dist(0.0f, 1.0f);

  // Weight mutation (inline)
  if (eneat::get_rand(mutate_or_not_mutate) <
      g.mutation_rates.connection_mutate_chance) {
    exfloat step = mutation_rates.step_size;
    for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
      if (eneat::get_rand(weight_dist) < mutation_rates.perturb_chance)
        it->second.weight += eneat::get_rand(weight_dist) * step * 2.0f - step;
      else
        it->second.weight = eneat::get_rand(weight_dist) * 4.0f - 2.0f;
    }
  }

  // Activation mutation (inline)
  if (eneat::get_rand(mutate_or_not_mutate) <
      g.mutation_rates.activation_mutation_chance) {
    for (auto it = g.genes.begin(); it != g.genes.end(); it++)
      it->second.activation = (ai_func_type)eneat::get_rand(act_dist);
  }

  // Link mutation uses direct sync access
  exfloat p = g.mutation_rates.link_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p) {
      // Simplified link mutation for init - just add a random connection
      std::uniform_int_distribution<size_t> dist1(0, g.max_neuron - 1);
      std::uniform_int_distribution<size_t> dist2(
          network_info.input_size + network_info.bias_size, g.max_neuron - 1);
      size_t n1 = eneat::get_rand(dist1);
      size_t n2 = eneat::get_rand(dist2);
      if (n1 != n2 && !is_output(n1, network_info)) {
        gene new_gene;
        new_gene.from_node = n1;
        new_gene.to_node = n2;
        new_gene.innovation_num =
            innovation_chan.add_gene_direct(n1, n2);
        new_gene.weight = eneat::get_rand(weight_dist) * 4.0f - 2.0f;
        new_gene.activation = (ai_func_type)eneat::get_rand(act_dist);
        g.genes[new_gene.innovation_num] = new_gene;
      }
    }
    p -= 1.0f;
  }

  // Neuron mutation uses direct sync access
  p = g.mutation_rates.neuron_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p && g.genes.size() > 0) {
      g.max_neuron++;
      std::uniform_int_distribution<size_t> dist(0, g.genes.size() - 1);
      size_t gene_id = eneat::get_rand(dist);
      auto it = g.genes.begin();
      std::advance(it, gene_id);
      if (it->second.enabled) {
        it->second.enabled = false;
        gene new_gene1;
        new_gene1.from_node = it->second.from_node;
        new_gene1.to_node = g.max_neuron - 1;
        new_gene1.weight = eneat::get_rand(weight_dist);
        new_gene1.activation = (ai_func_type)eneat::get_rand(act_dist);
        new_gene1.innovation_num = innovation_chan.add_gene_direct(
            new_gene1.from_node, new_gene1.to_node);
        new_gene1.enabled = true;
        gene new_gene2;
        new_gene2.from_node = g.max_neuron - 1;
        new_gene2.to_node = it->second.to_node;
        new_gene2.weight = it->second.weight;
        new_gene2.activation = it->second.activation;
        new_gene2.innovation_num = innovation_chan.add_gene_direct(
            new_gene2.from_node, new_gene2.to_node);
        new_gene2.enabled = true;
        g.genes[new_gene1.innovation_num] = new_gene1;
        g.genes[new_gene2.innovation_num] = new_gene2;
      }
    }
    p -= 1.0f;
  }

  // Bias neuron mutation (sync version)
  p = g.mutation_rates.bias_neuron_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p) {
      // Create a new bias neuron
      size_t new_neuron_id = g.max_neuron++;

      // Build set of existing bias neuron IDs from genes
      std::unordered_set<size_t> bias_nodes;
      for (const auto &[_, gene] : g.genes) {
        if (gene.is_bias_source) {
          bias_nodes.insert(gene.from_node);
        }
      }

      // Connect to a random hidden/output neuron
      if (g.max_neuron > network_info.input_size + network_info.bias_size + 1) {
        std::uniform_int_distribution<size_t> dist(
            network_info.input_size + network_info.bias_size, g.max_neuron - 2);
        size_t target = eneat::get_rand(dist);

        // Check if target is an evolved bias neuron
        if (!bias_nodes.count(target)) {
          gene new_gene;
          new_gene.from_node = new_neuron_id;
          new_gene.to_node = target;
          new_gene.is_bias_source = true;  // Mark as bias source
          new_gene.innovation_num =
              innovation_chan.add_gene_direct(new_neuron_id, target);
          new_gene.weight = eneat::get_rand(weight_dist) * 4.0f - 2.0f;
          new_gene.activation = (ai_func_type)eneat::get_rand(act_dist);
          g.genes[new_gene.innovation_num] = new_gene;
        }
      }
    }
    p -= 1.0f;
  }

  // Enable/disable mutations (inline, no innovation needed)
  p = g.mutation_rates.enable_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p) {
      std::vector<gene *> v;
      for (auto it = g.genes.begin(); it != g.genes.end(); it++)
        if (!it->second.enabled)
          v.push_back(&it->second);
      if (v.size() > 0) {
        std::uniform_int_distribution<int> dist(0, v.size() - 1);
        v[eneat::get_rand(dist)]->enabled = true;
      }
    }
    p -= 1.0f;
  }

  p = g.mutation_rates.disable_mutation_chance;
  while (p > 0.0f) {
    if (eneat::get_rand(mutate_or_not_mutate) < p) {
      std::vector<gene *> v;
      for (auto it = g.genes.begin(); it != g.genes.end(); it++)
        if (it->second.enabled)
          v.push_back(&it->second);
      if (v.size() > 0) {
        std::uniform_int_distribution<int> dist(0, v.size() - 1);
        v[eneat::get_rand(dist)]->enabled = false;
      }
    }
    p -= 1.0f;
  }
}

// Disjoint genes: genes present in one genome but not the other,
// within the range of both genomes' innovation numbers
exfloat pool::disjoint(const genome &g1, const genome &g2) {
  if (g1.genes.empty() || g2.genes.empty())
    return 0.0f;

  // Get max innovation number in each genome
  size_t max_innov1 = g1.genes.rbegin()->second.innovation_num;
  size_t max_innov2 = g2.genes.rbegin()->second.innovation_num;
  size_t min_max = std::min(max_innov1, max_innov2);

  size_t disjoint_count = 0;
  auto it1 = g1.genes.begin();
  auto it2 = g2.genes.begin();

  while (it1 != g1.genes.end() && it2 != g2.genes.end()) {
    size_t innov1 = it1->second.innovation_num;
    size_t innov2 = it2->second.innovation_num;

    if (innov1 == innov2) {
      ++it1;
      ++it2;
    } else if (innov1 < innov2) {
      // Gene in g1 not in g2, check if within range
      if (innov1 <= min_max)
        disjoint_count++;
      ++it1;
    } else {
      // Gene in g2 not in g1, check if within range
      if (innov2 <= min_max)
        disjoint_count++;
      ++it2;
    }
  }

  // Don't count remaining genes as disjoint - they're excess
  return static_cast<exfloat>(disjoint_count);
}

// Excess genes: genes beyond the max innovation number of the other genome
exfloat pool::excess(const genome &g1, const genome &g2) {
  if (g1.genes.empty() || g2.genes.empty())
    return static_cast<exfloat>(g1.genes.size() + g2.genes.size());

  size_t max_innov1 = g1.genes.rbegin()->second.innovation_num;
  size_t max_innov2 = g2.genes.rbegin()->second.innovation_num;

  size_t excess_count = 0;

  // Count genes in g1 beyond g2's max innovation
  for (auto it = g1.genes.rbegin(); it != g1.genes.rend(); ++it) {
    if (it->second.innovation_num > max_innov2)
      excess_count++;
    else
      break;
  }

  // Count genes in g2 beyond g1's max innovation
  for (auto it = g2.genes.rbegin(); it != g2.genes.rend(); ++it) {
    if (it->second.innovation_num > max_innov1)
      excess_count++;
    else
      break;
  }

  return static_cast<exfloat>(excess_count);
}

// Mutation number difference (rtNEAT-style):
// Average difference in mutation_num for matching genes
exfloat pool::mut_diff(const genome &g1, const genome &g2) {
  exfloat sum = 0.0f;
  size_t coincident = 0;

  for (auto it1 = g1.genes.begin(); it1 != g1.genes.end(); ++it1) {
    auto it2 = g2.genes.find(it1->second.innovation_num);
    if (it2 != g2.genes.end()) {
      coincident++;
      sum += std::abs(it1->second.mutation_num - it2->second.mutation_num);
    }
  }

  if (coincident == 0)
    return 0.0f;

  return sum / static_cast<exfloat>(coincident);
}

// rtNEAT-style compatibility distance:
// delta_disjoint * D + delta_excess * E + delta_weights * W
bool pool::is_same_species(const genome &g1, const genome &g2) {
  const exfloat dd = speciating_parameters.delta_disjoint * disjoint(g1, g2);
  const exfloat de = speciating_parameters.delta_excess * excess(g1, g2);
  const exfloat dw = speciating_parameters.delta_weights * mut_diff(g1, g2);
  return dd + de + dw < speciating_parameters.delta_threshold;
}

void pool::rank_globally() {
  std::vector<genome *> global;
  for (auto s = species.begin(); s != species.end(); s++) {
    s->genomes.modify([&](std::vector<genome> &g) {
      for (size_t i = 0; i < g.size(); i++)
        global.push_back(&g[i]);
    });
  }
  std::sort(global.begin(), global.end(), [](genome *&a, genome *&b) -> bool {
    if (a->fitness.load() == b->fitness.load())
      return a->genes.size() < b->genes.size();
    else
      return a->fitness.load() < b->fitness.load();
  });
  for (size_t j = 0; j < global.size(); j++)
    global[j]->global_rank.store(j + 1);
}

void pool::calculate_average_fitness(specie &s) {
  s.genomes.modify([&](std::vector<genome> &g) {
    size_t total = 0;
    size_t genome_count = g.size();
    for (size_t i = 0; i < genome_count; i++)
      total += g[i].global_rank.load();
    if (genome_count > 0)
      s.average_fitness.store(total / genome_count);
  });
}

size_t pool::total_average_fitness() {
  size_t total = 0;
  for (auto s = species.begin(); s != species.end(); s++)
    total += (*s).average_fitness.load();
  return total;
}

void pool::cull_species(const bool &cut_to_one) {
  for (auto s = species.begin(); s != species.end(); s++) {
    s->genomes.modify([&](std::vector<genome> &g) {
      std::sort(g.begin(), g.end(),
                [](genome &a, genome &b) { return a.fitness.load() > b.fitness.load(); });
      size_t remaining = std::ceil(g.size() * 1.0f / 2.0f);
      if (cut_to_one)
        remaining = 1;
      while (g.size() > remaining)
        g.pop_back();
      g.shrink_to_fit();  // Reclaim memory from culled genomes
    });
  }
}

ethreads::coro_task<genome> pool::breed_child(specie &s) {
  genome child(network_info, mutation_rates);
  std::uniform_real_distribution<exfloat> distributor(0.0f, 1.0f);
  bool should_return_early = false;

  // Access genomes via sync_shared_value to select parents
  s.genomes.modify([&](std::vector<genome> &g) {
    if (g.empty()) {
      // No genomes to breed from, return empty child
      should_return_early = true;
      return;
    }
    std::uniform_int_distribution<size_t> choose_genome(0, g.size() - 1);
    if (eneat::get_rand(distributor) < mutation_rates.crossover_chance) {
      // Copy genomes for crossover (to release lock before mutation)
      genome g1 = g[eneat::get_rand(choose_genome)];
      genome g2 = g[eneat::get_rand(choose_genome)];
      child = crossover(g1, g2);
    } else {
      child = g[eneat::get_rand(choose_genome)];
    }
  });

  if (should_return_early) {
    co_return child;
  }

  // Mutate the child asynchronously (outside the lock)
  co_await mutate(child);
  co_return child;
}

void pool::remove_stale_species() {
  auto s = species.begin();
  while (s != species.end()) {
    size_t best_fitness = 0;
    bool is_empty = false;
    s->genomes.modify([&](std::vector<genome> &g) {
      if (g.empty()) {
        is_empty = true;
        return;
      }
      genome &best = *(std::max_element(
          g.begin(), g.end(),
          [](genome &a, genome &b) -> bool { return a.fitness.load() < b.fitness.load(); }));
      best_fitness = best.fitness.load();
    });
    if (is_empty) {
      species.erase(s++);
      continue;
    }
    if (best_fitness > s->top_fitness.load()) {
      s->top_fitness.store(best_fitness);
      s->staleness.store(0);
    } else
      s->staleness.modify([](size_t &v) { ++v; });
    // NOTE: This is removing a species when its higher than the set maximum
    // fitness...
    if (!(s->staleness.load() < speciating_parameters.stale_species ||
          s->top_fitness.load() >= max_fitness.load()))
      species.erase(s++);
    else
      s++;
  }
}

void pool::remove_weak_species() {
  size_t sum = total_average_fitness();
  auto s = species.begin();
  while (s != species.end()) {
    exfloat breed = std::floor(1. * s->average_fitness.load() / (1. * sum) * 1. *
                               speciating_parameters.population);
    if (breed >= 1.0f)
      s++;
    else
      species.erase(s++);
  }
}

void pool::add_to_species(const genome &child) {
  auto s = species.begin();
  while (s != species.end()) {
    bool added = false;
    s->genomes.modify([&](std::vector<genome> &g) {
      if (!g.empty() && is_same_species(child, g[0])) {
        g.push_back(child);
        added = true;
      }
    });
    if (added) {
      return;
    }
    s++;
  }

  // No matching species found, create new one
  specie new_specie;
  new_specie.genomes.modify([&](std::vector<genome> &g) {
    g.push_back(child);
  });
  species.push_back(new_specie);
}

void pool::init_species_channel() {
  species_chan = std::make_unique<eneat::species_channel>(
      species,
      [this](const genome &g1, const genome &g2) { return is_same_species(g1, g2); });
}

ethreads::coro_task<void> pool::add_to_species_async(genome child) {
  if (species_chan) {
    species_chan->request_add_sync(std::move(child));
  } else {
    // Fallback to sync if channel not initialized
    add_to_species(child);
  }
  co_return;
}

// Async version of cull_species - sorts genomes in each species in parallel
ethreads::coro_task<void> pool::cull_species_async(const bool &cut_to_one) {
  // Create tasks to cull each species in parallel
  std::vector<ethreads::coro_task<void>> cull_tasks;

  for (auto &s : species) {
    cull_tasks.push_back(
        [](specie &sp, bool cut) -> ethreads::coro_task<void> {
          // Access genomes via sync_shared_value
          sp.genomes.modify([&](std::vector<genome> &g) {
            std::sort(g.begin(), g.end(),
                      [](genome &a, genome &b) { return a.fitness.load() > b.fitness.load(); });
            size_t remaining = std::ceil(g.size() * 1.0f / 2.0f);
            if (cut)
              remaining = 1;
            while (g.size() > remaining)
              g.pop_back();
            g.shrink_to_fit();  // Reclaim memory from culled genomes
          });
          co_return;
        }(s, cut_to_one));
  }

  co_await ethreads::when_all(std::move(cull_tasks));
}

// Async version of rank_globally - with yield points for cooperative scheduling
ethreads::coro_task<void> pool::rank_globally_async() {
  std::vector<genome *> global;

  // Collect pointers to all genomes via sync_shared_value
  for (auto &s : species) {
    s.genomes.modify([&](std::vector<genome> &g) {
      for (size_t i = 0; i < g.size(); i++)
        global.push_back(&g[i]);
    });
  }

  // Sort all genomes by fitness
  std::sort(global.begin(), global.end(), [](genome *&a, genome *&b) -> bool {
    if (a->fitness.load() == b->fitness.load())
      return a->genes.size() < b->genes.size();
    else
      return a->fitness.load() < b->fitness.load();
  });

  // Yield to let other coroutines run after heavy sort operation
  co_await ethreads::yield();

  // Assign ranks (genome pointers are still valid since genomes vector hasn't changed)
  for (size_t j = 0; j < global.size(); j++)
    global[j]->global_rank.store(j + 1);

  co_return;
}

// Async version that calculates average fitness for all species in parallel
ethreads::coro_task<void> pool::calculate_all_average_fitness_async() {
  std::vector<ethreads::coro_task<void>> fitness_tasks;

  for (auto &s : species) {
    fitness_tasks.push_back(
        [](specie &sp) -> ethreads::coro_task<void> {
          // Access genomes via sync_shared_value
          sp.genomes.modify([&](std::vector<genome> &g) {
            size_t total = 0;
            size_t genome_count = g.size();
            for (size_t i = 0; i < genome_count; i++)
              total += g[i].global_rank.load();
            if (genome_count > 0)
              sp.average_fitness.store(total / genome_count);
          });
          co_return;
        }(s));
  }

  co_await ethreads::when_all(std::move(fitness_tasks));
}

// Fully async version of new_generation using when_all for parallel operations
ethreads::coro_task<void> pool::new_generation_async() {
  // Reset innovation tracking for this generation
  innovation_chan.reset();

  // Initialize species channel for concurrent child addition
  init_species_channel();

  // Phase 1: Parallel species prep
  co_await cull_species_async(false);
  co_await rank_globally_async();

  // Sequential: modifies species list
  remove_stale_species();

  // Phase 2: Parallel fitness calculation
  co_await calculate_all_average_fitness_async();

  // Sequential: modifies species list
  remove_weak_species();

  // Calculate breeding counts BEFORE culling (uses average_fitness)
  size_t sum = total_average_fitness();
  std::vector<std::pair<specie*, size_t>> breed_counts;
  for (auto &s : species) {
    size_t breed = std::floor(((1. * s.average_fitness.load()) / (1. * sum)) * 1. *
                              speciating_parameters.population) -
                   1;
    breed_counts.push_back({&s, breed});
  }

  // Cull to one genome per species BEFORE breeding (so breed_child sees consistent state)
  co_await cull_species_async(true);

  // Now create breed_child tasks (species have 1 genome each now)
  std::vector<ethreads::coro_task<genome>> children;
  // Pre-reserve to avoid reallocation during push_back
  size_t total_breed = 0;
  for (const auto& [sp, count] : breed_counts) {
    total_breed += count;
  }
  children.reserve(total_breed + speciating_parameters.population);

  for (const auto& [sp, breed] : breed_counts) {
    for (size_t i = 0; i < breed; i++)
      children.push_back(breed_child(*sp));
  }

  std::uniform_int_distribution<size_t> choose_specie(0, species.size() - 1);
  std::vector<specie *> species_pointer;
  species_pointer.reserve(species.size());
  for (auto &s : species)
    species_pointer.push_back(&s);

  if (species.size() > 0) {
    while (children.size() + species.size() < speciating_parameters.population)
      children.push_back(
          breed_child(*species_pointer[eneat::get_rand(choose_specie)]));
  }

  // Create service tasks (don't spawn detached - we need to await them)
  auto innovation_service = innovation_chan.run_service();
  innovation_service.start();

  // Start all child breeding tasks in parallel
  for (auto &child_task : children) {
    child_task.start();
  }

  // Create species service task
  auto species_service = species_chan->run_service();
  species_service.start();

  // Use when_all to collect all children in parallel
  auto child_genomes = co_await ethreads::when_all(std::move(children));

  // Add children to species via channel (async with when_all)
  std::vector<ethreads::coro_task<void>> add_tasks;
  add_tasks.reserve(child_genomes.size());
  for (auto &child : child_genomes) {
    add_tasks.push_back(species_chan->request_add_async(std::move(child)));
  }
  co_await ethreads::when_all(std::move(add_tasks));

  // Stop services and wait for them to complete
  innovation_chan.stop();
  species_chan->stop();
  co_await innovation_service;  // Wait for innovation service to fully exit
  co_await species_service;     // Wait for species service to fully exit

  // Cleanup channel
  species_chan.reset();

  generation_number.modify([](size_t &v) { ++v; });

  co_return;
}

void pool::new_generation() {
  // Reset innovation tracking for this generation
  innovation_chan.reset();

  // Initialize species channel for concurrent child addition
  init_species_channel();

  cull_species(false);
  rank_globally();
  remove_stale_species();
  for (auto s = species.begin(); s != species.end(); s++)
    calculate_average_fitness(*s);
  remove_weak_species();

  // Calculate breeding counts BEFORE culling (uses average_fitness)
  size_t sum = total_average_fitness();
  std::vector<std::pair<specie*, size_t>> breed_counts;
  for (auto &s : species) {
    size_t breed = std::floor(((1. * s.average_fitness.load()) / (1. * sum)) * 1. *
                              speciating_parameters.population) -
                   1;
    breed_counts.push_back({&s, breed});
  }

  // Cull to one genome per species BEFORE breeding (so breed_child sees consistent state)
  cull_species(true);

  // Now create breed_child tasks (species have 1 genome each now)
  std::vector<ethreads::coro_task<genome>> children;
  // Pre-reserve to avoid reallocation during push_back
  size_t total_breed = 0;
  for (const auto& [sp, count] : breed_counts) {
    total_breed += count;
  }
  children.reserve(total_breed + speciating_parameters.population);

  for (const auto& [sp, breed] : breed_counts) {
    for (size_t i = 0; i < breed; i++)
      children.push_back(breed_child(*sp));
  }

  std::uniform_int_distribution<size_t> choose_specie(0, species.size() - 1);
  std::vector<specie *> species_pointer;
  species_pointer.reserve(species.size());
  for (auto &s : species)
    species_pointer.push_back(&s);

  if (species.size() == 0) {
    // everyone is dead
  } else
    while (children.size() + species.size() < speciating_parameters.population)
      children.push_back(
          breed_child(*species_pointer[eneat::get_rand(choose_specie)]));

  // Create service tasks (don't spawn detached - we need to wait for them)
  auto innovation_service = innovation_chan.run_service();
  innovation_service.start();

  // Start all child breeding tasks
  for (auto &child_task : children) {
    child_task.start();
  }

  // Create species service task
  auto species_service = species_chan->run_service();
  species_service.start();

  // Collect children and add to species via channel
  // The channel service processes requests concurrently
  for (auto &child_task : children) {
    species_chan->request_add_sync(child_task.get());
  }

  // Stop services and wait for them to complete
  innovation_chan.stop();
  species_chan->stop();
  innovation_service.get();  // Wait for innovation service to fully exit
  species_service.get();     // Wait for species service to fully exit

  // Cleanup channel
  species_chan.reset();

  generation_number.modify([](size_t &v) { ++v; });
}

// ============================================================================
// rtNEAT Methods - Real-time evolution support
// ============================================================================

// rtNEAT: Adjust fitness with age penalties and bonuses
void pool::adjust_species_fitness(specie &s) {
  s.genomes.modify([&](std::vector<genome> &gs) {
    for (auto &g : gs) {
      exfloat fitness = static_cast<exfloat>(g.fitness.load());

      // Age penalty for stagnant species
      if (s.age > speciating_parameters.dropoff_age) {
        size_t age_debt = s.age - s.age_of_last_improvement;
        if (age_debt >= speciating_parameters.dropoff_age) {
          // Extreme penalty for very old species without improvement
          fitness *= 0.01f;
        }
      }

      // Age bonus for young species (first 10 generations)
      if (s.age <= 10) {
        fitness *= speciating_parameters.age_significance;
      }

      // Ensure fitness doesn't go negative
      if (fitness < 0.0001f) fitness = 0.0001f;

      // Fitness sharing: divide by species size
      fitness /= static_cast<exfloat>(gs.size());

      g.adjusted_fitness.store(static_cast<size_t>(fitness));
    }
  });
}

// rtNEAT: Roulette wheel species selection based on average_est
specie* pool::choose_parent_species() {
  if (species.empty()) return nullptr;

  // Calculate total average estimate
  size_t total = 0;
  for (auto &s : species) {
    total += s.average_est.load();
  }

  if (total == 0) {
    // If no estimates, choose randomly
    std::uniform_int_distribution<size_t> dist(0, species.size() - 1);
    auto it = species.begin();
    std::advance(it, eneat::get_rand(dist));
    return &(*it);
  }

  // Roulette wheel selection
  std::uniform_int_distribution<size_t> dist(0, total - 1);
  size_t marble = eneat::get_rand(dist);
  size_t accumulated = 0;

  for (auto &s : species) {
    accumulated += s.average_est.load();
    if (accumulated > marble) {
      return &s;
    }
  }

  // Fallback to last species
  return &species.back();
}

// rtNEAT: Estimate running average for each species (only mature organisms)
void pool::estimate_all_averages() {
  for (auto &s : species) {
    size_t sum = 0;
    size_t count = 0;

    // Load genomes and iterate (read-only access)
    auto genomes_copy = s.genomes.load();
    for (const auto &g : genomes_copy) {
      if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
        sum += g.fitness.load();
        count++;
      }
    }

    if (count > 0) {
      s.average_est.store(sum / count);
    } else {
      s.average_est.store(0);
    }
  }
}

// rtNEAT: Age all organisms (increment time_alive)
void pool::age_all_organisms() {
  for (auto &s : species) {
    s.genomes.modify([](std::vector<genome> &gs) {
      for (auto &g : gs) {
        g.age();
      }
    });
    s.increment_age();
  }
}

// rtNEAT: Remove worst organism (lowest adjusted fitness among mature organisms)
ethreads::coro_task<void> pool::remove_worst_async() {
  specie* worst_species = nullptr;
  size_t worst_idx = 0;
  size_t min_fitness = std::numeric_limits<size_t>::max();

  // Find worst organism among mature ones
  for (auto &s : species) {
    auto genomes_copy = s.genomes.load();
    for (size_t i = 0; i < genomes_copy.size(); i++) {
      const auto &g = genomes_copy[i];
      if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
        size_t adj = g.adjusted_fitness.load();
        if (adj < min_fitness) {
          min_fitness = adj;
          worst_species = &s;
          worst_idx = i;
        }
      }
    }
  }

  // Remove worst organism
  if (worst_species) {
    worst_species->genomes.modify([worst_idx](std::vector<genome> &gs) {
      if (worst_idx < gs.size()) {
        gs.erase(gs.begin() + static_cast<std::ptrdiff_t>(worst_idx));
      }
    });

    // Remove empty species
    species.remove_if([](const specie &s) {
      return s.genomes.load().empty();
    });
  }

  co_return;
}

// rtNEAT: Produce single offspring (continuous evolution)
ethreads::coro_task<genome> pool::reproduce_one_async() {
  // Choose parent species based on fitness
  specie* parent_species = choose_parent_species();

  if (!parent_species) {
    // No species - return empty genome
    co_return genome(network_info, mutation_rates);
  }

  // Breed a child from the chosen species
  genome child = co_await breed_child(*parent_species);

  // Apply mutations
  co_await mutate(child);

  // Reset time_alive for new organism
  child.time_alive.store(0);

  co_return child;
}

// rtNEAT: Trait mutation - perturb random trait's parameters
ethreads::coro_task<void> pool::mutate_random_trait(genome &g) {
  if (g.traits.empty()) co_return;

  std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size() - 1);
  size_t trait_idx = eneat::get_rand(trait_dist);

  g.traits[trait_idx].mutate(
      mutation_rates.trait_param_mutation_power,
      0.5f  // 50% chance per parameter
  );

  co_return;
}

// rtNEAT: Link trait mutation - change gene's trait assignment
ethreads::coro_task<void> pool::mutate_link_trait(genome &g) {
  if (g.genes.empty() || g.traits.empty()) co_return;

  std::uniform_int_distribution<size_t> gene_dist(0, g.genes.size() - 1);
  std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size());  // 0 = no trait

  size_t gene_idx = eneat::get_rand(gene_dist);
  auto it = g.genes.begin();
  std::advance(it, gene_idx);

  it->second.trait_id = eneat::get_rand(trait_dist);

  co_return;
}

std::istream &operator>>(std::istream &input, pool &p) {
  size_t innovation_num;
  input >> innovation_num;
  p.innovation_chan.set_innovation_number(innovation_num);
  size_t gen_num, max_fit;
  input >> gen_num;
  input >> max_fit;
  p.generation_number.store(gen_num);
  p.max_fitness.store(max_fit);
  input >> p.network_info.input_size >> p.network_info.output_size >>
      p.network_info.bias_size;
  p.network_info.functional_neurons = p.network_info.input_size +
                                      p.network_info.output_size +
                                      p.network_info.bias_size;
  input >> p.network_info.recurrent;
  input >> p.speciating_parameters;
  input >> p.mutation_rates;
  size_t species_number;
  input >> species_number;
  p.species.clear();
  for (size_t c = 0; c < species_number; c++) {
    specie new_specie;
    size_t tmp_top_fitness, tmp_average_fitness, tmp_staleness;
    input >> tmp_top_fitness;
    input >> tmp_average_fitness;
    input >> tmp_staleness;
    new_specie.top_fitness.store(tmp_top_fitness);
    new_specie.average_fitness.store(tmp_average_fitness);
    new_specie.staleness.store(tmp_staleness);
    size_t specie_population;
    input >> specie_population;
    for (size_t i = 0; i < specie_population; i++) {
      genome new_genome(p.network_info, p.mutation_rates);
      size_t tmp_fitness, tmp_adjusted_fitness, tmp_global_rank;
      input >> tmp_fitness;
      input >> tmp_adjusted_fitness;
      input >> tmp_global_rank;
      new_genome.fitness.store(tmp_fitness);
      new_genome.adjusted_fitness.store(tmp_adjusted_fitness);
      new_genome.global_rank.store(tmp_global_rank);
      input >> new_genome.mutation_rates;
      size_t gene_number;
      input >> new_genome.max_neuron >> gene_number;
      for (size_t j = 0; j < gene_number; j++) {
        gene new_gene;
        input >> new_gene.innovation_num;
        input >> new_gene.from_node;
        input >> new_gene.to_node;
        input >> new_gene.weight;
        input >> new_gene.enabled;
        int activation;
        input >> activation;
        new_gene.activation = (ai_func_type)activation;
        input >> new_gene.is_bias_source;
        new_genome.genes[new_gene.innovation_num] = new_gene;
      }

      new_specie.genomes.modify([&](std::vector<genome> &g) {
        g.push_back(new_genome);
      });
    }

    p.species.push_back(new_specie);
  }
  return input;
}

std::ostream &operator<<(std::ostream &output, pool &p) {
  output << p.innovation_chan.number() << std::endl;
  output << p.generation_number.load() << std::endl;
  output << p.max_fitness.load() << std::endl;
  output << p.network_info.input_size << " " << p.network_info.output_size
         << " " << p.network_info.bias_size << std::endl;
  output << p.network_info.recurrent << std::endl;
  p.network_info.functional_neurons = p.network_info.input_size +
                                      p.network_info.output_size +
                                      p.network_info.bias_size;
  output << p.speciating_parameters;
  output << p.mutation_rates;
  output << p.species.size() << std::endl;
  for (auto &sp : p.species) {
    output << "   ";
    output << sp.top_fitness.load() << " ";
    output << sp.average_fitness.load() << " ";
    output << sp.staleness.load() << std::endl;
    auto genomes_copy = sp.genomes.load();
    output << "   " << genomes_copy.size() << std::endl;
    for (size_t i = 0; i < genomes_copy.size(); i++) {
      output << "      ";
      output << genomes_copy[i].fitness.load() << " ";
      output << genomes_copy[i].adjusted_fitness.load() << " ";
      output << genomes_copy[i].global_rank.load() << std::endl;
      output << genomes_copy[i].mutation_rates;
      output << "      " << genomes_copy[i].max_neuron << " "
             << genomes_copy[i].genes.size() << std::endl;
      for (auto pair : genomes_copy[i].genes) {
        gene &g = pair.second;
        output << "         ";
        output << g.innovation_num << " " << g.from_node << " " << g.to_node
               << " " << g.weight << " " << g.enabled << " " << g.activation
               << " " << g.is_bias_source << std::endl;
      }
    }

    output << std::endl << std::endl;
  }
  return output;
}
