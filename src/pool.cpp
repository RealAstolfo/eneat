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

pool::pool(size_t input, size_t output, size_t population, size_t bias, bool rec) {
  network_info.input_size = input;
  network_info.output_size = output;
  network_info.bias_size = bias;
  network_info.functional_neurons = input + output + bias;
  network_info.recurrent = rec;
  speciating_parameters.population = population;  // Set before creating genomes
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

// Safe index-based genome access functions

std::vector<std::pair<size_t, size_t>> pool::get_genome_indices() {
  std::vector<std::pair<size_t, size_t>> result;
  size_t species_idx = 0;
  for (auto& s : species) {
    size_t genome_count = s.genome_count();
    for (size_t i = 0; i < genome_count; i++) {
      result.push_back({species_idx, i});
    }
    species_idx++;
  }
  return result;
}

size_t pool::get_population_size() {
  size_t count = 0;
  for (const auto& s : species) {
    count += s.genome_count();
  }
  return count;
}

genome pool::get_genome_copy(size_t species_idx, size_t genome_idx) {
  auto it = species.begin();
  std::advance(it, species_idx);
  if (it == species.end()) {
    return genome(network_info, mutation_rates);  // Return empty genome if invalid
  }
  genome result(network_info, mutation_rates);
  it->genomes.modify([&](std::vector<genome>& g) {
    if (genome_idx < g.size()) {
      result = g[genome_idx];
    }
  });
  return result;
}

void pool::set_genome_fitness(size_t species_idx, size_t genome_idx, size_t fitness) {
  auto it = species.begin();
  std::advance(it, species_idx);
  if (it == species.end()) return;
  it->genomes.modify([&](std::vector<genome>& g) {
    if (genome_idx < g.size()) {
      g[genome_idx].fitness.store(fitness);
    }
  });

  // Check if this is a new best genome
  genome copy = get_genome_copy(species_idx, genome_idx);
  update_best_genome(copy);
}

void pool::increment_genome_age(size_t species_idx, size_t genome_idx) {
  auto it = species.begin();
  std::advance(it, species_idx);
  if (it == species.end()) return;
  it->genomes.modify([&](std::vector<genome>& g) {
    if (genome_idx < g.size()) {
      g[genome_idx].time_alive.store(g[genome_idx].time_alive.load() + 1);
    }
  });
}

// Default crossover - delegates to multipoint
genome pool::crossover(const genome &g1, const genome &g2) {
  // rtNEAT: Select mating method based on probabilities
  std::uniform_real_distribution<exfloat> dist(0.0f, 1.0f);
  exfloat roll = eneat::get_rand(dist);

  // multipoint_avg_chance and singlepoint_chance determine the probabilities
  // Remaining probability goes to multipoint
  exfloat multipoint_prob = 1.0f - mutation_rates.multipoint_avg_chance - mutation_rates.singlepoint_chance;

  if (roll < multipoint_prob) {
    return crossover_multipoint(g1, g2);
  } else if (roll < multipoint_prob + mutation_rates.multipoint_avg_chance) {
    return crossover_multipoint_avg(g1, g2);
  } else {
    return crossover_singlepoint(g1, g2);
  }
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

    // Clamp weight to rtNEAT limits (genome.cpp:1262-1263)
    if (it->second.weight > 8.0f) it->second.weight = 8.0f;
    else if (it->second.weight < -8.0f) it->second.weight = -8.0f;

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
  // rtNEAT: recur_only_prob - force recurrent links only
  std::uniform_real_distribution<exfloat> prob_dist(0.0f, 1.0f);
  bool force_recurrent = g.can_be_recurrent &&
                         eneat::get_rand(prob_dist) < mutation_rates.recur_only_prob;

  // Retry loop for finding valid connection (rtNEAT newlink_tries)
  for (size_t attempt = 0; attempt < speciating_parameters.newlink_tries; attempt++) {
    std::uniform_int_distribution<size_t> distributor1(0, g.max_neuron - 1);
    size_t neuron1 = eneat::get_rand(distributor1);
    std::uniform_int_distribution<size_t> distributor2(
        network_info.input_size + network_info.bias_size, g.max_neuron - 1);
    size_t neuron2 = eneat::get_rand(distributor2);

    // Skip invalid combinations and retry
    if (is_output(neuron1, network_info) && is_output(neuron2, network_info))
      continue;
    if (is_bias(neuron2, network_info))
      continue;
    if (neuron1 == neuron2 && (!force_bias))
      continue;
    if (is_output(neuron1, network_info))
      std::swap(neuron1, neuron2);
    if (force_bias) {
      std::uniform_int_distribution<size_t> bias_choose(
          network_info.input_size,
          network_info.input_size + network_info.output_size - 1);
      neuron1 = eneat::get_rand(bias_choose);
    }

    // Check for recurrence in non-recurrent networks
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
        continue;  // Retry instead of returning
    }

    // rtNEAT: recur_only_prob - if force_recurrent, skip non-recurrent connections
    if (force_recurrent) {
      // Check if this connection would be recurrent (self-loop or creates cycle)
      bool is_recurrent = (neuron1 == neuron2);  // Self-loop is always recurrent

      if (!is_recurrent && !is_bias(neuron1, network_info) && !is_input(neuron1, network_info)) {
        // Check if adding neuron1->neuron2 creates a cycle
        std::unordered_map<size_t, std::vector<size_t>> connections;
        for (const auto &gene : g.genes) {
          if (gene.second.enabled) {
            connections[gene.second.from_node].push_back(gene.second.to_node);
          }
        }
        connections[neuron1].push_back(neuron2);

        std::queue<size_t> que;
        que.push(neuron2);
        std::unordered_set<size_t> visited;
        visited.insert(neuron2);

        while (!que.empty()) {
          size_t tmp = que.front();
          que.pop();
          if (tmp == neuron1) {
            is_recurrent = true;
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

      if (!is_recurrent)
        continue;  // Skip non-recurrent connections when force_recurrent
    }

    // Check if connection already exists
    bool already_exists = false;
    for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
      if (it->second.from_node == neuron1 && it->second.to_node == neuron2) {
        already_exists = true;
        break;
      }
    }
    if (already_exists)
      continue;  // Retry instead of returning

    // Found valid connection - add it
    gene new_gene;
    new_gene.from_node = neuron1;
    new_gene.to_node = neuron2;

    // Direct innovation tracking (no channel service needed for rtNEAT)
    new_gene.innovation_num = innovation_chan.add_gene_direct(
        new_gene.from_node, new_gene.to_node);
    std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> act_generator(ai_func_type::FIRST,
                                                        ai_func_type::LAST);
    // rtNEAT uses [-1, 1] range (genome.cpp:1758)
    new_gene.weight = eneat::get_rand(weight_generator) * 2.0f - 1.0f;
    new_gene.activation = (ai_func_type)eneat::get_rand(act_generator);
    g.genes[new_gene.innovation_num] = new_gene;
    co_return;  // Success - exit after adding gene
  }
  // Failed to find valid connection after all attempts - silently return
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
  // Direct innovation tracking (no channel service needed for rtNEAT)
  new_gene1.innovation_num = innovation_chan.add_gene_direct(
      new_gene1.from_node, new_gene1.to_node);
  new_gene1.enabled = true;
  gene new_gene2;
  new_gene2.from_node = g.max_neuron - 1;
  new_gene2.to_node = it->second.to_node;
  new_gene2.weight = it->second.weight;
  new_gene2.activation = it->second.activation;
  // Direct innovation tracking (no channel service needed for rtNEAT)
  new_gene2.innovation_num = innovation_chan.add_gene_direct(
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
  // Direct innovation tracking (no channel service needed for rtNEAT)
  new_gene.innovation_num = innovation_chan.add_gene_direct(
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
  // Self-adaptive mutation rates disabled to match rtNEAT reference behavior
  // (Rates are now static as defined in mutation_rate_container defaults)

  std::uniform_real_distribution<exfloat> mutate_or_not_mutate(0.0f, 1.0f);

  // rtNEAT: Track if structural mutation occurred
  // Trait mutations are only performed if no structural mutation happened
  bool structural_mutation = false;
  size_t genes_before = g.genes.size();

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

  // Check if structural mutation occurred (genes added)
  if (g.genes.size() > genes_before) {
    structural_mutation = true;
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

  // rtNEAT: Trait mutations only if no structural mutation occurred
  if (!structural_mutation) {
    // Trait mutations (Hebbian learning parameters)
    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.trait_mutation_chance)
      co_await mutate_random_trait(g);

    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.link_trait_mutation_chance)
      co_await mutate_link_trait(g);

    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.node_trait_mutation_chance)
      co_await mutate_node_trait(g);
  }

  co_return;
}

// Synchronous mutation for single-threaded initialization
void pool::mutate_sync(genome &g) {
  // Self-adaptive mutation rates disabled to match rtNEAT reference behavior
  // (Rates are now static as defined in mutation_rate_container defaults)

  std::uniform_real_distribution<exfloat> mutate_or_not_mutate(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> act_dist(ai_func_type::FIRST,
                                                 ai_func_type::LAST);
  std::uniform_real_distribution<exfloat> weight_dist(0.0f, 1.0f);

  // rtNEAT: Track if structural mutation occurred
  size_t genes_before = g.genes.size();

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

  // rtNEAT: Trait mutations only if no structural mutation occurred
  bool structural_mutation = (g.genes.size() > genes_before);

  if (!structural_mutation) {
    // Trait mutations (Hebbian learning parameters) - sync versions
    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.trait_mutation_chance) {
      if (!g.traits.empty()) {
        std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size() - 1);
        size_t trait_idx = eneat::get_rand(trait_dist);
        g.traits[trait_idx].mutate(mutation_rates.trait_param_mutation_power, 0.5f);
      }
    }

    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.link_trait_mutation_chance) {
      if (!g.genes.empty() && !g.traits.empty()) {
        std::uniform_int_distribution<size_t> gene_dist(0, g.genes.size() - 1);
        std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size());  // 0 = no trait
        size_t gene_idx = eneat::get_rand(gene_dist);
        auto it = g.genes.begin();
        std::advance(it, gene_idx);
        it->second.trait_id = eneat::get_rand(trait_dist);
        // rtNEAT: Update mutation_num for speciation distance calculation
        it->second.mutation_num += 1.0f;
      }
    }

    // Node trait mutation (sync version)
    if (eneat::get_rand(mutate_or_not_mutate) < g.mutation_rates.node_trait_mutation_chance) {
      if (!g.traits.empty()) {
        // Collect all unique node IDs
        std::set<size_t> nodes;
        for (const auto& [_, gene] : g.genes) {
          nodes.insert(gene.from_node);
          nodes.insert(gene.to_node);
        }
        if (!nodes.empty()) {
          std::uniform_int_distribution<size_t> node_dist(0, nodes.size() - 1);
          size_t node_idx = eneat::get_rand(node_dist);
          auto node_it = nodes.begin();
          std::advance(node_it, node_idx);
          size_t node_id = *node_it;

          // Check if node is frozen
          if (g.frozen_nodes.count(node_id) == 0) {
            std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size());
            g.node_traits[node_id] = eneat::get_rand(trait_dist);
          }
        }
      }
    }
  }
}

// Weight-only mutation for super champion offspring
// rtNEAT reference: 80% of super champion offspring get weight mutations only
void pool::mutate_weight_only(genome &g) {
  std::uniform_real_distribution<exfloat> dist(0.0f, 1.0f);
  exfloat step = mutation_rates.step_size;

  // 50% chance of severe mutation mode (higher mutation rates)
  bool severe = eneat::get_rand(dist) > 0.5f;
  exfloat gausspoint = severe ? 0.3f : 0.1f;
  exfloat coldgausspoint = severe ? 0.1f : 0.05f;

  // Track gene position for position-based mutation power
  size_t gene_total = g.genes.size();
  size_t endpart = static_cast<size_t>(gene_total * 0.8f);
  size_t num = 0;

  for (auto& [_, gene] : g.genes) {
    // Skip frozen genes
    if (gene.frozen) {
      num++;
      continue;
    }

    // Newer genes (after 80% mark) get more aggressive mutation
    exfloat power_mod = (num > endpart) ? 1.5f : 1.0f;
    exfloat effective_step = step * power_mod;

    exfloat roll = eneat::get_rand(dist);
    if (roll < gausspoint) {
      // Gaussian: perturb existing weight
      gene.weight += eneat::get_rand(dist) * effective_step * 2.0f - effective_step;
    } else if (roll < gausspoint + coldgausspoint) {
      // Cold Gaussian: replace weight entirely
      gene.weight = eneat::get_rand(dist) * 4.0f - 2.0f;
    }
    // else: no change to this gene

    // Clamp weight to rtNEAT limits (genome.cpp:1262-1263)
    if (gene.weight > 8.0f) gene.weight = 8.0f;
    else if (gene.weight < -8.0f) gene.weight = -8.0f;

    gene.mutation_num += 1.0f;
    num++;
  }
}

// rtNEAT: mutate_add_sensor - connects unconnected sensors to outputs
// This finds input sensors that aren't connected to all outputs and adds connections
// Reference: genome.cpp:1804
ethreads::coro_task<void> pool::mutate_add_sensor(genome &g) {
  mutate_add_sensor_sync(g);
  co_return;
}

void pool::mutate_add_sensor_sync(genome &g) {
  std::uniform_real_distribution<exfloat> weight_dist(-3.0f, 3.0f);
  std::uniform_int_distribution<size_t> trait_dist(0, std::max((size_t)1, g.traits.size()) - 1);

  // Find all input sensor node IDs
  std::vector<size_t> sensor_ids;
  for (size_t i = 0; i < network_info.input_size; i++) {
    sensor_ids.push_back(i);
  }

  // Find all output node IDs
  std::vector<size_t> output_ids;
  size_t output_start = network_info.input_size + network_info.bias_size;
  for (size_t i = 0; i < network_info.output_size; i++) {
    output_ids.push_back(output_start + i);
  }

  // For each sensor, count how many outputs it's connected to
  // and eliminate sensors that are already fully connected
  std::vector<size_t> available_sensors;
  for (size_t sensor_id : sensor_ids) {
    size_t output_connections = 0;
    for (const auto& [_, gene] : g.genes) {
      if (gene.from_node == sensor_id && gene.enabled) {
        // Check if to_node is an output
        for (size_t out_id : output_ids) {
          if (gene.to_node == out_id) {
            output_connections++;
            break;
          }
        }
      }
    }
    // If not connected to all outputs, this sensor is available
    if (output_connections < output_ids.size()) {
      available_sensors.push_back(sensor_id);
    }
  }

  // If all sensors are fully connected, nothing to do
  if (available_sensors.empty()) {
    return;
  }

  // Pick a random sensor from available ones
  std::uniform_int_distribution<size_t> sensor_choice(0, available_sensors.size() - 1);
  size_t chosen_sensor = available_sensors[eneat::get_rand(sensor_choice)];

  // Add connections from chosen sensor to any outputs not already connected
  for (size_t output_id : output_ids) {
    // Check if connection already exists
    bool exists = false;
    for (const auto& [_, gene] : g.genes) {
      if (gene.from_node == chosen_sensor && gene.to_node == output_id) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      // Get innovation number for this connection
      size_t innov = innovation_chan.add_gene_direct(chosen_sensor, output_id);

      // Create new gene
      gene new_gene;
      new_gene.innovation_num = innov;
      new_gene.from_node = chosen_sensor;
      new_gene.to_node = output_id;
      new_gene.weight = eneat::get_rand(weight_dist);
      new_gene.enabled = true;
      new_gene.is_recurrent = false;
      new_gene.mutation_num = 0.0f;

      // Assign random trait if available
      if (!g.traits.empty()) {
        new_gene.trait_id = eneat::get_rand(trait_dist) + 1;  // trait_id is 1-indexed
      }

      g.genes[innov] = new_gene;
    }
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

void pool::calculate_average_fitness(specie &s) {
  s.genomes.modify([&](std::vector<genome> &g) {
    size_t total = 0;
    size_t genome_count = g.size();
    for (size_t i = 0; i < genome_count; i++)
      total += g[i].fitness.load();
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

  // Mutate the child synchronously (avoids thread pool starvation from nested co_await)
  mutate_sync(child);
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
  // Protect against edge cases that would remove all species
  if (species.empty()) return;

  size_t sum = total_average_fitness();

  // Don't remove species if total fitness is 0 (would remove all)
  if (sum == 0) return;

  // Always keep at least one species
  size_t min_species = 1;

  auto s = species.begin();
  while (s != species.end() && species.size() > min_species) {
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
  // rtNEAT adds one child at a time - use direct synchronous add
  // (channel is unnecessary overhead for single additions)
  add_to_species(child);
  co_return;
}

// ============================================================================
// rtNEAT Methods - Real-time evolution support
// ============================================================================

// rtNEAT: Adjust fitness with age penalties and bonuses
// Based on rtNEAT species.cpp:696-741
void pool::adjust_species_fitness(specie &s) {
  s.genomes.modify([&](std::vector<genome> &gs) {
    // Calculate age_debt using rtNEAT formula
    // age_debt = (age - age_of_last_improvement + 1) - dropoff_age
    int64_t age_debt = static_cast<int64_t>(s.age) -
                       static_cast<int64_t>(s.age_of_last_improvement) + 1 -
                       static_cast<int64_t>(speciating_parameters.dropoff_age);

    // rtNEAT: if age_debt == 0, set to 1 to ensure penalty is applied
    if (age_debt == 0) age_debt = 1;

    for (auto &g : gs) {
      exfloat fitness = static_cast<exfloat>(g.fitness.load());

      // Remember original fitness (rtNEAT: org->orig_fitness = org->fitness)
      g.orig_fitness = g.fitness.load();

      // rtNEAT: Apply penalty if age_debt >= 1 OR obliterate is set
      // This means species that haven't improved since dropoff_age get penalized
      if (age_debt >= 1 || s.obliterate) {
        // Extreme penalty for stagnation (divide fitness by 100)
        fitness *= 0.01f;
      }

      // Age bonus for young species (first 10 generations)
      // rtNEAT: age_significance multiplier (1.0 = no boost)
      if (s.age <= 10) {
        fitness *= speciating_parameters.age_significance;
      }

      // Do not allow negative fitness
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

    // Zero-copy iteration
    s.genomes.modify([&](const std::vector<genome>& gs) {
      for (const auto &g : gs) {
        if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
          sum += g.fitness.load();
          count++;
        }
      }
    });

    if (count > 0) {
      s.average_est.store(sum / count);
    } else {
      // rtNEAT fallback: if no mature organisms, use ALL organisms (species.cpp:152-156)
      size_t fallback_sum = 0;
      size_t fallback_count = 0;
      s.genomes.modify([&](const std::vector<genome>& gs) {
        for (const auto &g : gs) {
          fallback_sum += g.fitness.load();
          fallback_count++;
        }
      });
      if (fallback_count > 0) {
        s.average_est.store(fallback_sum / fallback_count);
      } else {
        s.average_est.store(0);
      }
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
// Returns true if an organism was removed, false if no mature organisms found
ethreads::coro_task<bool> pool::remove_worst_async() {
  // Use index-based tracking to avoid pointer invalidation
  size_t worst_species_idx = std::numeric_limits<size_t>::max();
  size_t worst_genome_idx = 0;
  size_t min_fitness = std::numeric_limits<size_t>::max();

  // Find worst organism among mature ones (zero-copy iteration)
  size_t species_idx = 0;
  for (auto &s : species) {
    s.genomes.modify([&](const std::vector<genome>& gs) {
      for (size_t i = 0; i < gs.size(); i++) {
        const auto &g = gs[i];
        if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
          size_t adj = g.adjusted_fitness.load();
          if (adj < min_fitness) {
            min_fitness = adj;
            worst_species_idx = species_idx;
            worst_genome_idx = i;
          }
        }
      }
    });
    species_idx++;
  }

  // Remove worst organism using index-based access
  if (worst_species_idx != std::numeric_limits<size_t>::max()) {
    // Re-fetch the species iterator to ensure it's valid
    auto it = species.begin();
    std::advance(it, worst_species_idx);

    if (it != species.end()) {
      it->genomes.modify([worst_genome_idx](std::vector<genome> &gs) {
        if (worst_genome_idx < gs.size()) {
          gs.erase(gs.begin() + static_cast<std::ptrdiff_t>(worst_genome_idx));
        }
      });

      // Remove empty species
      species.remove_if([](const specie &s) {
        return s.is_empty();
      });

      co_return true;  // Successfully removed an organism
    }
  }

  co_return false;  // No mature organisms found to remove
}

// rtNEAT: Produce single offspring (continuous evolution)
ethreads::coro_task<genome> pool::reproduce_one_async() {
  // Choose parent species based on fitness
  specie* parent_species = choose_parent_species();

  if (!parent_species) {
    // No species - return empty genome
    co_return genome(network_info, mutation_rates);
  }

  // Breed a child from the chosen species (includes mutation via mutate_sync)
  genome child = co_await breed_child(*parent_species);

  // Reset time_alive for new organism
  child.time_alive.store(0);

  co_return child;
}

// ============================================================================
// Synchronous versions of rtNEAT functions (avoid coroutine scheduling overhead)
// ============================================================================

genome pool::breed_child_sync(specie &s) {
  genome child(network_info, mutation_rates);
  std::uniform_real_distribution<exfloat> distributor(0.0f, 1.0f);

  // Access genomes via sync_shared_value to select parents
  bool should_return_early = false;
  bool crossover_happened = false;

  s.genomes.modify([&](std::vector<genome> &g) {
    if (g.empty()) {
      should_return_early = true;
      return;
    }
    std::uniform_int_distribution<size_t> choose_genome(0, g.size() - 1);

    if (eneat::get_rand(distributor) < mutation_rates.crossover_chance) {
      genome g1 = g[eneat::get_rand(choose_genome)];
      genome g2;

      // rtNEAT: Interspecies mating check
      if (eneat::get_rand(distributor) < speciating_parameters.interspecies_mate_rate &&
          species.size() > 1) {
        // Mate with champion from random other species
        specie* other = choose_random_species_excluding(s);
        if (other) {
          g2 = get_species_champion(*other);
        } else {
          g2 = g[eneat::get_rand(choose_genome)];
        }
      } else {
        g2 = g[eneat::get_rand(choose_genome)];
      }

      child = crossover(g1, g2);
      crossover_happened = true;
    } else {
      child = g[eneat::get_rand(choose_genome)];
    }
  });

  if (should_return_early) {
    return child;
  }

  // Track origin (for debugging/adaptive strategies)
  child.mate_baby = crossover_happened;

  // rtNEAT: Mate-only probability - skip mutation with mate_only_prob chance after crossover
  bool should_mutate = true;
  if (crossover_happened) {
    std::uniform_real_distribution<exfloat> dist(0.0f, 1.0f);
    if (eneat::get_rand(dist) < mutation_rates.mutate_only_prob) {
      // Note: mutate_only_prob is inverted here - high value = more mutation skipping after mating
      // This matches rtNEAT reference where mate_only_prob controls skipping mutation
      should_mutate = false;
    }
  }

  if (should_mutate) {
    mutate_sync(child);
    // Track if structural mutation happened
    child.mut_struct_baby = (child.genes.size() > 0);  // Simplified check
  }

  return child;
}

genome pool::reproduce_one() {
  specie* parent_species = choose_parent_species();

  if (!parent_species) {
    return genome(network_info, mutation_rates);
  }

  genome child(network_info, mutation_rates);

  // rtNEAT: Champion preservation - clone champion once per generation if species has enough offspring
  if (parent_species->expected_offspring.load() > 5 && !parent_species->champion_preserved) {
    parent_species->champion_preserved = true;
    child = get_species_champion(*parent_species);
    child.fitness.store(0);
    child.adjusted_fitness.store(0);
    child.time_alive.store(0);
    child.mate_baby = false;
    child.mut_struct_baby = false;
    // Champion clone - no mutation
    return child;
  }

  // rtNEAT: Super champion offspring handling
  // Check if parent species has a super champion with reserved offspring
  auto genomes_copy = parent_species->genomes.load();
  for (auto& g : genomes_copy) {
    if (g.super_champ_offspring > 0) {
      // Clone the super champion
      child = g;
      child.fitness.store(0);
      child.adjusted_fitness.store(0);
      child.time_alive.store(0);
      child.mate_baby = false;

      // Decrement super_champ_offspring (need to modify the original)
      parent_species->genomes.modify([&](std::vector<genome>& gs) {
        for (auto& genome_ref : gs) {
          if (genome_ref.pop_champ && genome_ref.super_champ_offspring > 0) {
            genome_ref.super_champ_offspring--;
            break;
          }
        }
      });

      // 80% chance: weight mutations only, 20%: full mutation
      std::uniform_real_distribution<exfloat> dist(0.0f, 1.0f);
      if (eneat::get_rand(dist) < 0.8f) {
        // Weight mutation only (with severe mode and position-based power)
        mutate_weight_only(child);
        child.mut_struct_baby = false;
      } else {
        mutate_sync(child);
        child.mut_struct_baby = true;
      }
      return child;
    }
  }

  // Normal breeding
  child = breed_child_sync(*parent_species);
  child.time_alive.store(0);
  return child;
}

bool pool::remove_worst() {
  size_t worst_species_idx = std::numeric_limits<size_t>::max();
  size_t worst_genome_idx = 0;
  size_t min_fitness = std::numeric_limits<size_t>::max();

  size_t species_idx = 0;
  for (auto &s : species) {
    s.genomes.modify([&](const std::vector<genome>& gs) {
      for (size_t i = 0; i < gs.size(); i++) {
        const auto &g = gs[i];
        if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
          size_t adj = g.adjusted_fitness.load();
          if (adj < min_fitness) {
            min_fitness = adj;
            worst_species_idx = species_idx;
            worst_genome_idx = i;
          }
        }
      }
    });
    species_idx++;
  }

  if (worst_species_idx != std::numeric_limits<size_t>::max()) {
    auto it = species.begin();
    std::advance(it, worst_species_idx);

    if (it != species.end()) {
      it->genomes.modify([worst_genome_idx](std::vector<genome> &gs) {
        if (worst_genome_idx < gs.size()) {
          gs.erase(gs.begin() + static_cast<std::ptrdiff_t>(worst_genome_idx));
        }
      });

      species.remove_if([](const specie &s) {
        return s.is_empty();
      });

      return true;
    }
  }

  return false;
}

// rtNEAT: Probabilistic worst removal - fitness-proportional selection for removal
// Reference: population.cpp probabilistic removal variant
// Lower fitness = higher probability of being selected for removal
bool pool::remove_worst_probabilistic() {
  // Collect all eligible organisms with their locations and inverse fitness weights
  struct candidate {
    size_t species_idx;
    size_t genome_idx;
    exfloat inverse_fitness;  // Higher = more likely to be removed
  };
  std::vector<candidate> candidates;
  exfloat total_inverse = 0.0f;

  size_t species_idx = 0;
  for (auto &s : species) {
    auto genomes_copy = s.genomes.load();
    for (size_t i = 0; i < genomes_copy.size(); i++) {
      const auto &g = genomes_copy[i];
      if (g.time_alive.load() >= speciating_parameters.time_alive_minimum) {
        size_t adj = g.adjusted_fitness.load();
        // Inverse fitness: lower fitness = higher weight for removal
        // Add 1 to avoid division by zero
        exfloat inv = 1.0f / (static_cast<exfloat>(adj) + 1.0f);
        candidates.push_back({species_idx, i, inv});
        total_inverse += inv;
      }
    }
    species_idx++;
  }

  if (candidates.empty() || total_inverse <= 0.0f) {
    return false;
  }

  // Weighted random selection
  std::uniform_real_distribution<exfloat> dist(0.0f, total_inverse);
  exfloat selection = eneat::get_rand(dist);

  exfloat cumulative = 0.0f;
  size_t selected_species_idx = candidates[0].species_idx;
  size_t selected_genome_idx = candidates[0].genome_idx;

  for (const auto &c : candidates) {
    cumulative += c.inverse_fitness;
    if (cumulative >= selection) {
      selected_species_idx = c.species_idx;
      selected_genome_idx = c.genome_idx;
      break;
    }
  }

  // Remove the selected organism
  auto it = species.begin();
  std::advance(it, selected_species_idx);

  if (it != species.end()) {
    it->genomes.modify([selected_genome_idx](std::vector<genome> &gs) {
      if (selected_genome_idx < gs.size()) {
        gs.erase(gs.begin() + static_cast<std::ptrdiff_t>(selected_genome_idx));
      }
    });

    // Remove empty species
    species.remove_if([](const specie &s) {
      return s.is_empty();
    });

    return true;
  }

  return false;
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

  // rtNEAT: Update mutation_num for speciation distance calculation
  it->second.mutation_num += 1.0f;

  co_return;
}

// rtNEAT: Mutate node trait - assign a random trait to a random node
ethreads::coro_task<void> pool::mutate_node_trait(genome &g) {
  if (g.traits.empty()) co_return;

  // Collect all unique node IDs from genes
  std::set<size_t> nodes;
  for (const auto& [_, gene] : g.genes) {
    nodes.insert(gene.from_node);
    nodes.insert(gene.to_node);
  }

  if (nodes.empty()) co_return;

  // Choose a random node
  std::uniform_int_distribution<size_t> node_dist(0, nodes.size() - 1);
  size_t node_idx = eneat::get_rand(node_dist);
  auto node_it = nodes.begin();
  std::advance(node_it, node_idx);
  size_t node_id = *node_it;

  // Check if node is frozen
  if (g.frozen_nodes.count(node_id) > 0) co_return;

  // Assign a random trait to this node (1-indexed, or 0 for no trait)
  std::uniform_int_distribution<size_t> trait_dist(0, g.traits.size());
  g.node_traits[node_id] = eneat::get_rand(trait_dist);

  co_return;
}

// rtNEAT: Check for population stagnation and trigger delta-coding if needed
void pool::check_delta_coding() {
  if (!speciating_parameters.delta_coding_enabled) return;

  size_t current_best = max_fitness.load();

  if (current_best > highest_fitness_ever) {
    highest_fitness_ever = current_best;
    generations_since_improvement = 0;
  } else {
    generations_since_improvement++;
  }

  // Trigger delta-coding if stagnated for dropoff_age + 5 generations
  if (generations_since_improvement >= speciating_parameters.dropoff_age + 5) {
    apply_delta_coding();
    generations_since_improvement = 0;
  }
}

// rtNEAT: Apply delta-coding - replace population with mutated clones of top 2 organisms
void pool::apply_delta_coding() {
  if (species.empty()) return;

  // Find top 2 organisms across all species
  // Store copies directly to avoid dangling pointers
  std::optional<genome> best_copy;
  std::optional<genome> second_copy;
  size_t best_fitness = 0;
  size_t second_fitness = 0;

  for (auto& s : species) {
    auto genomes_copy = s.genomes.load();
    for (const auto& g : genomes_copy) {
      size_t f = g.fitness.load();
      if (f > best_fitness) {
        second_copy = best_copy;
        second_fitness = best_fitness;
        best_copy = g;
        best_fitness = f;
      } else if (f > second_fitness) {
        second_copy = g;
        second_fitness = f;
      }
    }
  }

  if (!best_copy) return;

  // Use second_copy if available, otherwise use best_copy
  genome second = second_copy ? *second_copy : *best_copy;

  // Clear all species
  species.clear();

  // Create half population from best, half from second best
  size_t half_pop = speciating_parameters.population / 2;
  size_t total = speciating_parameters.population;

  for (size_t i = 0; i < half_pop; i++) {
    genome child = *best_copy;
    child.fitness.store(0);
    child.adjusted_fitness.store(0);
    child.time_alive.store(0);
    mutate_sync(child);
    add_to_species(child);
  }

  for (size_t i = half_pop; i < total; i++) {
    genome child = second;
    child.fitness.store(0);
    child.adjusted_fitness.store(0);
    child.time_alive.store(0);
    mutate_sync(child);
    add_to_species(child);
  }
}

// rtNEAT: Redistribute offspring from weak species to strong species
void pool::redistribute_offspring() {
  if (speciating_parameters.babies_stolen == 0) return;
  if (species.size() < 4) return;  // Need at least 4 species for meaningful redistribution

  // Sort species by average fitness (descending)
  std::vector<specie*> sorted_species;
  for (auto& s : species) {
    sorted_species.push_back(&s);
  }

  std::sort(sorted_species.begin(), sorted_species.end(),
    [](const specie* a, const specie* b) {
      return a->average_est.load() > b->average_est.load();
    });

  // Steal from weak species (bottom, age > 5, expected_offspring > 2)
  size_t stolen = 0;
  size_t target = speciating_parameters.babies_stolen;

  for (auto it = sorted_species.rbegin();
       it != sorted_species.rend() && stolen < target; ++it) {
    specie* s = *it;
    if (s->age > 5) {
      size_t available = s->expected_offspring.load();
      if (available > 2) {
        size_t to_steal = std::min(available - 1, target - stolen);
        s->expected_offspring.store(available - to_steal);
        stolen += to_steal;
      }
    }
  }

  if (stolen == 0) return;

  // Give stolen babies to top 3 species (proportionally: 2/5, 2/5, 1/5)
  size_t top3_count = std::min((size_t)3, sorted_species.size());
  if (top3_count >= 1) {
    size_t current = sorted_species[0]->expected_offspring.load();
    sorted_species[0]->expected_offspring.store(current + (stolen * 2) / 5);
  }
  if (top3_count >= 2) {
    size_t current = sorted_species[1]->expected_offspring.load();
    sorted_species[1]->expected_offspring.store(current + (stolen * 2) / 5);
  }
  if (top3_count >= 3) {
    size_t current = sorted_species[2]->expected_offspring.load();
    sorted_species[2]->expected_offspring.store(current + stolen - (stolen * 4) / 5);
  }
}

// rtNEAT: Adjust compatibility threshold to maintain target species count
void pool::adjust_compatibility_threshold() {
  if (!speciating_parameters.dynamic_threshold_enabled) return;

  offspring_since_compat_adjust++;

  if (offspring_since_compat_adjust >= speciating_parameters.compat_adjust_frequency) {
    offspring_since_compat_adjust = 0;

    size_t num_species = species.size();

    if (num_species < speciating_parameters.target_species_count) {
      // Too few species - decrease threshold to split species
      speciating_parameters.delta_threshold -=
          speciating_parameters.compat_threshold_delta;
    } else if (num_species > speciating_parameters.target_species_count) {
      // Too many species - increase threshold to merge species
      speciating_parameters.delta_threshold +=
          speciating_parameters.compat_threshold_delta;
    }

    // Enforce minimum threshold
    if (speciating_parameters.delta_threshold <
        speciating_parameters.compat_threshold_min) {
      speciating_parameters.delta_threshold =
          speciating_parameters.compat_threshold_min;
    }

    // Reassign all organisms to species with new threshold
    reassign_all_species();
  }
}

// rtNEAT: Reassign all organisms to species based on current threshold
// Reference: population.cpp - preserves species representatives for continuity
void pool::reassign_all_species() {
  // Collect all non-representative genomes for reassignment
  std::vector<genome> genomes_to_reassign;

  for (auto& s : species) {
    auto genomes_copy = s.genomes.load();
    // Keep first genome as representative, collect rest for reassignment
    for (size_t i = 1; i < genomes_copy.size(); i++) {
      genomes_to_reassign.push_back(genomes_copy[i]);
    }
    // Clear species to just the representative
    s.genomes.modify([](std::vector<genome>& gs) {
      if (gs.size() > 1) {
        gs.erase(gs.begin() + 1, gs.end());
      }
    });
  }

  // Re-add all non-representative genomes (they match against existing species
  // representatives based on new threshold, or form new species)
  for (auto& g : genomes_to_reassign) {
    add_to_species(g);
  }

  // Remove any species that became empty (shouldn't happen but safety check)
  species.remove_if([](const specie& s) {
    return s.is_empty();
  });
}

// rtNEAT: Calculate expected offspring for each species proportionally
void pool::calculate_expected_offspring() {
  // First pass: calculate total adjusted fitness (using average_est)
  size_t total_avg = 0;
  for (auto& s : species) {
    total_avg += s.average_est.load();
  }

  if (total_avg == 0) {
    // If no fitness yet, distribute evenly
    size_t per_species = speciating_parameters.population / species.size();
    for (auto& s : species) {
      s.expected_offspring.store(per_species);
    }
    return;
  }

  // Second pass: assign expected offspring proportionally
  size_t total_assigned = 0;
  for (auto& s : species) {
    size_t expected = (s.average_est.load() * speciating_parameters.population) / total_avg;
    s.expected_offspring.store(expected);
    total_assigned += expected;
  }

  // Assign remainder to best species (due to integer division rounding)
  if (total_assigned < speciating_parameters.population && !species.empty()) {
    size_t best_avg = 0;
    specie* best_species = nullptr;
    for (auto& s : species) {
      if (s.average_est.load() > best_avg) {
        best_avg = s.average_est.load();
        best_species = &s;
      }
    }
    if (best_species) {
      best_species->expected_offspring.store(
          best_species->expected_offspring.load() +
          (speciating_parameters.population - total_assigned));
    }
  }
}

// rtNEAT: Reset champion_preserved flags for new generation
void pool::reset_champion_flags() {
  for (auto& s : species) {
    s.champion_preserved = false;
  }
}

// rtNEAT: Choose a random species excluding the given one (for interspecies mating)
specie* pool::choose_random_species_excluding(const specie& exclude) {
  if (species.size() <= 1) return nullptr;

  // Collect candidates with their fitness
  std::vector<std::pair<specie*, size_t>> candidates;
  for (auto& s : species) {
    if (&s != &exclude) {
      candidates.push_back({&s, s.average_est.load()});
    }
  }

  if (candidates.empty()) return nullptr;

  // rtNEAT: Sort by fitness (descending) for Gaussian-weighted selection
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Gaussian-weighted selection: bias towards better species
  // Use absolute value of Gaussian to get index (lower = better)
  std::normal_distribution<double> gauss(0.0, 1.0);
  double rand_val = std::abs(eneat::get_rand(gauss));

  // Map Gaussian value to index (clamp to valid range)
  // Higher Gaussian values = higher indices (worse species)
  size_t index = static_cast<size_t>(rand_val * candidates.size() / 3.0);
  if (index >= candidates.size()) {
    index = candidates.size() - 1;
  }

  return candidates[index].first;
}

// rtNEAT: Get the champion (best fitness) genome from a species
genome pool::get_species_champion(const specie& s) {
  auto genomes_copy = s.genomes.load();
  if (genomes_copy.empty()) {
    return genome(network_info, mutation_rates);  // Return empty genome
  }

  std::optional<genome> best;
  size_t best_fitness = 0;

  for (const auto& g : genomes_copy) {
    if (g.fitness.load() >= best_fitness) {
      best_fitness = g.fitness.load();
      best = g;
    }
  }

  return best ? *best : genomes_copy[0];
}

// rtNEAT: Calculate network depth (max path length from inputs to outputs)
// Useful for complexity tracking and determining network structure
size_t pool::calculate_network_depth(const genome& g) {
  if (g.genes.empty()) return 0;

  // Build adjacency list (from_node -> to_node connections)
  std::unordered_map<size_t, std::vector<size_t>> adj;
  std::unordered_set<size_t> all_nodes;

  for (const auto& [_, gene] : g.genes) {
    if (gene.enabled) {
      adj[gene.from_node].push_back(gene.to_node);
      all_nodes.insert(gene.from_node);
      all_nodes.insert(gene.to_node);
    }
  }

  if (all_nodes.empty()) return 0;

  // BFS from each input node to find max depth
  size_t max_depth = 0;
  size_t input_end = g.network_info.input_size + g.network_info.bias_size;

  for (size_t input_node = 0; input_node < input_end; input_node++) {
    if (adj.find(input_node) == adj.end()) continue;

    // BFS with depth tracking
    std::queue<std::pair<size_t, size_t>> q;  // (node, depth)
    std::unordered_set<size_t> visited;

    q.push({input_node, 0});
    visited.insert(input_node);

    while (!q.empty()) {
      auto [node, depth] = q.front();
      q.pop();

      max_depth = std::max(max_depth, depth);

      // Limit depth to prevent infinite loops in recurrent networks
      if (depth > 100) continue;

      for (size_t next : adj[node]) {
        if (visited.find(next) == visited.end()) {
          visited.insert(next);
          q.push({next, depth + 1});
        }
      }
    }
  }

  return max_depth;
}

// rtNEAT: Verify genome integrity (debug/validation)
bool pool::verify_genome(const genome& g) const {
  // Check for valid network_info
  if (g.network_info.input_size == 0 || g.network_info.output_size == 0) {
    return false;
  }

  // Check all genes have valid node references
  for (const auto& [innovation, gene] : g.genes) {
    // from_node should be less than max_neuron
    if (gene.from_node >= g.max_neuron) {
      return false;
    }
    // to_node should be less than max_neuron
    if (gene.to_node >= g.max_neuron) {
      return false;
    }
    // to_node shouldn't be an input node
    if (gene.to_node < g.network_info.input_size + g.network_info.bias_size) {
      // Unless it's a bias node connecting to itself (shouldn't happen)
      return false;
    }
    // Weight should be reasonable (not NaN or infinite)
    if (std::isnan(gene.weight) || std::isinf(gene.weight)) {
      return false;
    }
    // mutation_num should be non-negative
    if (gene.mutation_num < 0.0f) {
      return false;
    }
  }

  // Check trait references are valid
  for (const auto& [innovation, gene] : g.genes) {
    if (gene.trait_id > g.traits.size()) {
      return false;  // Invalid trait reference
    }
  }

  // Check node trait references are valid
  for (const auto& [node_id, trait_id] : g.node_traits) {
    if (trait_id > g.traits.size()) {
      return false;  // Invalid trait reference
    }
  }

  // Check max_neuron is at least functional_neurons
  if (g.max_neuron < g.network_info.functional_neurons) {
    return false;
  }

  return true;
}

// Best genome tracking - thread-safe access to all-time best performer

std::optional<genome> pool::get_best_genome() const {
  std::shared_lock lock(best_genome_mutex);
  return best_genome_ever;
}

void pool::update_best_genome(const genome& g) {
  size_t fitness = g.fitness.load();

  // Check current max without exclusive lock first (fast path)
  if (fitness <= max_fitness.load()) {
    return;
  }

  // Need exclusive lock to update
  std::unique_lock lock(best_genome_mutex);

  // Double-check after acquiring lock (another thread may have updated)
  if (fitness > max_fitness.load()) {
    max_fitness.store(fitness);
    best_genome_ever = g;  // Deep copy via genome copy constructor
  }
}

std::optional<brain> pool::get_best_brain() const {
  std::shared_lock lock(best_genome_mutex);
  if (!best_genome_ever) {
    return std::nullopt;
  }

  brain b;
  b = *best_genome_ever;  // Build brain from genome
  b.flush();              // Reset to clean state for deterministic evaluation
  return b;
}

// rtNEAT iteration step: remove -> estimate -> reproduce (call-based, decoupled from evaluation)
// Reference: nero_evolution.cpp evolveBrains()
// Key insight: rtNEAT does removal FIRST, then estimates averages, then reproduces
// Extended: Dynamic population sizing - expand when under target, contract when over
bool pool::iteration_step() {
  size_t current_pop = get_population_size();
  size_t target_pop = speciating_parameters.population;

  // Step 1: Population-aware removal
  // Under target: skip removal to allow growth
  // At target: remove 1 (standard rtNEAT)
  // Over target: aggressive removal (remove multiple)
  bool removed = false;
  size_t removals_needed = 0;

  if (current_pop > target_pop) {
    // Over target: aggressive contraction - remove extras plus one more
    removals_needed = (current_pop - target_pop) + 1;
    // Cap at 10% of population per iteration to avoid instability
    removals_needed = std::min(removals_needed, std::max(1UL, current_pop / 10));
    for (size_t i = 0; i < removals_needed; i++) {
      if (remove_worst()) {
        removed = true;
      } else {
        break;  // No more mature organisms to remove
      }
    }
  } else if (current_pop >= target_pop) {
    // At target: standard rtNEAT - remove 1
    removed = remove_worst();
  } else {
    // Under target: skip removal to allow population growth
    // Still need at least one mature organism to proceed with reproduction
    // Check if we have any evaluated genomes
    bool has_evaluated = false;
    for (const auto& s : species) {
      auto gs = s.genomes.load();  // Read-only copy
      for (const auto& g : gs) {
        if (g.time_alive.load() > 0) {
          has_evaluated = true;
          break;
        }
      }
      if (has_evaluated) break;
    }
    if (!has_evaluated) {
      return false;  // No evaluated organisms yet
    }
    removed = true;  // Pretend we removed to allow offspring production
  }

  if (!removed) {
    return false;  // No mature organisms to remove
  }

  // Step 2: Estimate all species averages (AFTER removal, before reproduction)
  estimate_all_averages();

  // Step 3: Periodic generational features (every 10 iterations)
  if (iteration_count_ % 10 == 0) {
    // Adjust fitness with age penalties and fitness sharing
    for (auto& s : species) {
      adjust_species_fitness(s);
    }

    // Calculate expected offspring proportionally
    calculate_expected_offspring();

    // Redistribute offspring from weak to strong species (babies stolen)
    redistribute_offspring();

    // Check for population stagnation and apply delta-coding if needed
    check_delta_coding();
  }

  // Step 4: Produce offspring
  // Under target: produce multiple offspring to accelerate growth
  // At/over target: produce 1 (standard rtNEAT)
  size_t offspring_count = 1;
  if (current_pop < target_pop) {
    // Accelerate growth: produce up to 5 offspring per iteration when under target
    offspring_count = std::min(5UL, target_pop - current_pop);
  }

  for (size_t i = 0; i < offspring_count; i++) {
    genome child = reproduce_one();
    add_to_species(std::move(child));
  }

  // Step 5: Track iteration count for periodic operations
  iteration_count_++;

  // Step 6: Periodic compatibility threshold adjustment + species reassignment
  size_t compat_frequency = std::max(1UL, get_population_size() / 10);
  if (iteration_count_ % compat_frequency == 0) {
    adjust_compatibility_threshold();
    // Note: adjust_compatibility_threshold already calls reassign_all_species() internally
  }

  // Step 7: Per-generation operations (once per population_size iterations)
  if (iteration_count_ % speciating_parameters.population == 0) {
    for (auto& s : species) {
      s.increment_age();
    }
    reset_champion_flags();
  }

  // Step 8: Periodic species cleanup (every 10 generations)
  size_t cleanup_period = speciating_parameters.population * 10;
  if (iteration_count_ > 0 && iteration_count_ % cleanup_period == 0) {
    remove_stale_species();
    remove_weak_species();
  }

  return true;
}

std::istream &operator>>(std::istream &input, pool &p) {
  size_t innovation_num;
  input >> innovation_num;
  p.innovation_chan.set_innovation_number(innovation_num);
  size_t reserved, max_fit;
  input >> reserved;  // Reserved for backwards compatibility (was generation_number)
  input >> max_fit;
  p.max_fitness.store(max_fit);

  // Deserialize best genome if present
  int has_best;
  input >> has_best;
  if (has_best) {
    genome best(p.network_info, p.mutation_rates);
    size_t tmp_fitness, tmp_adjusted_fitness, tmp_global_rank;
    input >> tmp_fitness >> tmp_adjusted_fitness >> tmp_global_rank;
    best.fitness.store(tmp_fitness);
    best.adjusted_fitness.store(tmp_adjusted_fitness);
    best.global_rank.store(tmp_global_rank);
    input >> best.mutation_rates;
    // Deserialize traits for best genome
    size_t trait_count;
    input >> trait_count;
    best.traits.resize(trait_count);
    for (size_t t = 0; t < trait_count; t++) {
      input >> best.traits[t];
    }
    size_t gene_number;
    input >> best.max_neuron >> gene_number;
    for (size_t j = 0; j < gene_number; j++) {
      gene new_gene;
      input >> new_gene.innovation_num >> new_gene.from_node >> new_gene.to_node
            >> new_gene.weight >> new_gene.enabled;
      int activation;
      input >> activation;
      new_gene.activation = (ai_func_type)activation;
      input >> new_gene.is_bias_source;
      input >> new_gene.trait_id;
      best.genes[new_gene.innovation_num] = new_gene;
    }
    p.best_genome_ever = best;
  } else {
    p.best_genome_ever.reset();
  }

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
      // Deserialize traits
      size_t trait_count;
      input >> trait_count;
      new_genome.traits.resize(trait_count);
      for (size_t t = 0; t < trait_count; t++) {
        input >> new_genome.traits[t];
      }
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
        input >> new_gene.trait_id;
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
  output << 0 << std::endl;  // Reserved for backwards compatibility (was generation_number)
  output << p.max_fitness.load() << std::endl;

  // Serialize best genome if present
  output << (p.best_genome_ever.has_value() ? 1 : 0) << std::endl;
  if (p.best_genome_ever) {
    genome best = *p.best_genome_ever;  // Copy (mutation_rates << needs non-const)
    output << best.fitness.load() << " " << best.adjusted_fitness.load() << " "
           << best.global_rank.load() << std::endl;
    output << best.mutation_rates;
    // Serialize traits for best genome
    output << best.traits.size() << std::endl;
    for (const auto& trait : best.traits) {
      output << trait << std::endl;
    }
    output << best.max_neuron << " " << best.genes.size() << std::endl;
    for (const auto& pair : best.genes) {
      const gene& g = pair.second;
      output << g.innovation_num << " " << g.from_node << " " << g.to_node << " "
             << g.weight << " " << g.enabled << " " << g.activation << " "
             << g.is_bias_source << " " << g.trait_id << std::endl;
    }
  }

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
      // Serialize traits
      output << "      " << genomes_copy[i].traits.size() << std::endl;
      for (const auto& trait : genomes_copy[i].traits) {
        output << "         " << trait << std::endl;
      }
      output << "      " << genomes_copy[i].max_neuron << " "
             << genomes_copy[i].genes.size() << std::endl;
      for (auto pair : genomes_copy[i].genes) {
        gene &g = pair.second;
        output << "         ";
        output << g.innovation_num << " " << g.from_node << " " << g.to_node
               << " " << g.weight << " " << g.enabled << " " << g.activation
               << " " << g.is_bias_source << " " << g.trait_id << std::endl;
      }
    }

    output << std::endl << std::endl;
  }
  return output;
}
