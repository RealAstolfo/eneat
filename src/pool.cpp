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
  std::vector<std::pair<specie *, genome *>> genomes;
  for (auto s = this->species.begin(); s != this->species.end(); s++)
    for (size_t i = 0; i < (*s).genomes.size(); i++)
      genomes.push_back(std::make_pair(&(*s), &((*s).genomes[i])));
  return genomes;
}

genome pool::crossover(const genome &g1, const genome &g2) {
  if (g2.fitness > g1.fitness)
    return crossover(g2, g1);
  genome child(network_info, mutation_rates);
  auto it1 = g1.genes.begin();
  std::uniform_int_distribution<int> coin_flip(1, 2);
  for (; it1 != g1.genes.end(); it1++) {
    auto it2 = g2.genes.find(it1->second.innovation_num);
    if (it2 != g2.genes.end()) {
      int coin = eneat::get_rand(coin_flip);
      if (coin == 2)
        child.genes[it1->second.innovation_num] = it2->second;
      else
        child.genes[it1->second.innovation_num] = it1->second;
    } else
      child.genes[it1->second.innovation_num] = it1->second;
  }

  child.max_neuron = std::max(g1.max_neuron, g2.max_neuron);
  return child;
}

ethreads::coro_task<void> pool::mutate_activation(genome &g) {
  std::uniform_int_distribution<size_t> distributor(ai_func_type::FIRST,
                                                    ai_func_type::LAST);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    it->second.activation = (ai_func_type)eneat::get_rand(distributor);
  co_return;
}

ethreads::coro_task<void> pool::mutate_weight(genome &g) {
  exfloat step = mutation_rates.step_size;
  std::uniform_real_distribution<exfloat> real_distributor(0.0f, 1.0f);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (eneat::get_rand(real_distributor) < mutation_rates.perturb_chance)
      it->second.weight +=
          eneat::get_rand(real_distributor) * step * 2.0f - step;
    else
      it->second.weight = eneat::get_rand(real_distributor) * 4.0f - 2.0f;
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
    if (is_bias(neuron1, network_info) || is_input(neuron1, network_info))
      has_recurrence = false;
    else {
      std::unordered_map<size_t, std::vector<size_t>> connections;
      for (const auto &gene : g.genes) {
        connections[gene.second.from_node].push_back(gene.second.to_node);
      }
      connections[neuron1].push_back(neuron2);

      std::queue<size_t> que;
      que.push(neuron1);
      std::unordered_set<size_t> visited;
      visited.insert(neuron1);
      has_recurrence = false;

      while (!que.empty()) {
        size_t tmp = que.front();
        que.pop();

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
  std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
  new_gene1.weight = eneat::get_rand(weight_generator);
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

exfloat pool::disjoint(const genome &g1, const genome &g2) {
  std::deque<size_t> genes1(g1.genes.size());
  std::deque<size_t> genes2(g2.genes.size());

  // std::map should be pre-sorted.
  std::transform(g1.genes.begin(), g1.genes.end(), genes1.begin(),
                 [](const auto &gene) { return gene.second.innovation_num; });

  std::transform(g2.genes.begin(), g2.genes.end(), genes2.begin(),
                 [](const auto &gene) { return gene.second.innovation_num; });

  size_t disjoint_count = 0;
  size_t i = 0;
  size_t j = 0;

  while (i < genes1.size() && j < genes2.size()) {
    if (genes1[i] < genes2[j]) {
      disjoint_count++;
      i++;
    } else if (genes1[i] > genes2[j]) {
      disjoint_count++;
      j++;
    } else {
      i++;
      j++;
    }
  }

  disjoint_count += genes1.size() - i + genes2.size() - j;
  size_t gene_size = std::max(genes1.size(), genes2.size());

  return static_cast<exfloat>(disjoint_count) / static_cast<exfloat>(gene_size);
}

exfloat pool::weights(const genome &g1, const genome &g2) {
  auto it1 = g1.genes.begin();
  exfloat sum = 0.0;
  size_t coincident = 0;
  for (; it1 != g1.genes.end(); it1++) {
    auto it2 = g2.genes.find((*it1).second.innovation_num);
    if (it2 != g2.genes.end()) {
      coincident++;
      sum += std::abs((*it1).second.weight - (*it2).second.weight);
    }
  }

  return 1.0f * sum / (1.0f * coincident);
}

bool pool::is_same_species(const genome &g1, const genome &g2) {
  const exfloat dd = speciating_parameters.delta_disjoint * disjoint(g1, g2);
  exfloat dw = speciating_parameters.delta_weights * weights(g1, g2);
  return dd + dw < speciating_parameters.delta_threshold;
}

void pool::rank_globally() {
  std::vector<genome *> global;
  for (auto s = species.begin(); s != species.end(); s++)
    for (size_t i = 0; i < s->genomes.size(); i++)
      global.push_back(&((*s).genomes[i]));
  std::sort(global.begin(), global.end(), [](genome *&a, genome *&b) -> bool {
    if (a->fitness == b->fitness)
      return a->genes.size() < b->genes.size();
    else
      return a->fitness < b->fitness;
  });
  for (size_t j = 0; j < global.size(); j++)
    global[j]->global_rank = j + 1;
}

void pool::calculate_average_fitness(specie &s) {
  size_t total = 0;
  for (size_t i = 0; i < s.genomes.size(); i++)
    total += s.genomes[i].global_rank;
  s.average_fitness = total / s.genomes.size();
}

size_t pool::total_average_fitness() {
  size_t total = 0;
  for (auto s = species.begin(); s != species.end(); s++)
    total += (*s).average_fitness;
  return total;
}

void pool::cull_species(const bool &cut_to_one) {
  for (auto s = species.begin(); s != species.end(); s++) {
    std::sort(s->genomes.begin(), s->genomes.end(),
              [](genome &a, genome &b) { return a.fitness > b.fitness; });
    size_t remaining = std::ceil(s->genomes.size() * 1.0f / 2.0f);
    if (cut_to_one)
      remaining = 1;
    while (s->genomes.size() > remaining)
      s->genomes.pop_back();
  }
}

ethreads::coro_task<genome> pool::breed_child(specie &s) {
  genome child(network_info, mutation_rates);
  std::uniform_real_distribution<exfloat> distributor(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> choose_genome(0, s.genomes.size() - 1);
  if (eneat::get_rand(distributor) < mutation_rates.crossover_chance) {
    genome &g1 = s.genomes[eneat::get_rand(choose_genome)];
    genome &g2 = s.genomes[eneat::get_rand(choose_genome)];
    child = crossover(g1, g2);
  } else {
    genome &g = s.genomes[eneat::get_rand(choose_genome)];
    child = g;
  }

  // Mutate the child asynchronously
  co_await mutate(child);
  co_return child;
}

void pool::remove_stale_species() {
  auto s = species.begin();
  while (s != species.end()) {
    genome &g = *(std::max_element(
        s->genomes.begin(), s->genomes.end(),
        [](genome &a, genome &b) -> bool { return a.fitness < b.fitness; }));
    if (g.fitness > s->top_fitness) {
      s->top_fitness = g.fitness;
      s->staleness = 0;
    } else
      s->staleness++;
    // NOTE: This is removing a species when its higher than the set maximum
    // fitness...
    if (!(s->staleness < speciating_parameters.stale_species ||
          s->top_fitness >= max_fitness))
      species.erase(s++);
    else
      s++;
  }
}

void pool::remove_weak_species() {
  size_t sum = total_average_fitness();
  auto s = species.begin();
  while (s != species.end()) {
    exfloat breed = std::floor(1. * s->average_fitness / (1. * sum) * 1. *
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
    if (is_same_species(child, s->genomes[0])) {
      s->genomes.push_back(child);
      return;
    }
    s++;
  }

  // No matching species found, create new one
  specie new_specie;
  new_specie.genomes.push_back(child);
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

  std::vector<ethreads::coro_task<genome>> children;
  size_t sum = total_average_fitness();
  for (auto s = species.begin(); s != species.end(); s++) {
    size_t breed = std::floor(((1. * s->average_fitness) / (1. * sum)) * 1. *
                              speciating_parameters.population) -
                   1;
    for (size_t i = 0; i < breed; i++)
      children.push_back(breed_child(*s));
  }

  cull_species(true);
  std::uniform_int_distribution<size_t> choose_specie(0, species.size() - 1);
  std::vector<specie *> species_pointer(0);
  for (auto s = species.begin(); s != species.end(); s++)
    species_pointer.push_back(&(*s));
  if (species.size() == 0) {
    // everyone is dead
  } else
    while (children.size() + species.size() < speciating_parameters.population)
      children.push_back(
          breed_child(*species_pointer[eneat::get_rand(choose_specie)]));

  // Spawn innovation service to process mutation requests
  ethreads::g_runtime.spawn_detached(innovation_chan.run_service());

  // Start all child breeding tasks
  for (auto &child_task : children) {
    child_task.start();
  }

  // Spawn species service to process add requests
  ethreads::g_runtime.spawn_detached(species_chan->run_service());

  // Collect children and add to species via channel
  // The channel service processes requests concurrently
  for (auto &child_task : children) {
    species_chan->request_add_sync(child_task.get());
  }

  // Stop services
  innovation_chan.stop();
  species_chan->stop();

  // Cleanup channel
  species_chan.reset();

  generation_number++;
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
    input >> new_specie.top_fitness;
    input >> new_specie.average_fitness;
    input >> new_specie.staleness;
    size_t specie_population;
    input >> specie_population;
    for (size_t i = 0; i < specie_population; i++) {
      genome new_genome(p.network_info, p.mutation_rates);
      input >> new_genome.fitness;
      input >> new_genome.adjusted_fitness;
      input >> new_genome.global_rank;
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
        new_genome.genes[new_gene.innovation_num] = new_gene;
      }

      new_specie.genomes.push_back(new_genome);
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
  for (auto specie : p.species) {
    output << "   ";
    output << specie.top_fitness << " ";
    output << specie.average_fitness << " ";
    output << specie.staleness << std::endl;
    output << "   " << specie.genomes.size() << std::endl;
    for (size_t i = 0; i < specie.genomes.size(); i++) {
      output << "      ";
      output << specie.genomes[i].fitness << " ";
      output << specie.genomes[i].adjusted_fitness << " ";
      output << specie.genomes[i].global_rank << std::endl;
      output << specie.genomes[i].mutation_rates;
      output << "      " << specie.genomes[i].max_neuron << " "
             << specie.genomes[i].genes.size() << std::endl;
      for (auto pair : specie.genomes[i].genes) {
        gene &g = pair.second;
        output << "         ";
        output << g.innovation_num << " " << g.from_node << " " << g.to_node
               << " " << g.weight << " " << g.enabled << " " << g.activation
               << std::endl;
      }
    }

    output << std::endl << std::endl;
  }
  return output;
}
