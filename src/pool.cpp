#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <random>
#include <stddef.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
  generator.seed(rd());
  for (size_t i = 0; i < speciating_parameters.population; i++) {
    genome new_genome(network_info, mutation_rates);
    mutate(new_genome);
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
      int coin = get_rand(coin_flip, generator);
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

void pool::mutate_activation(genome &g) {
  std::uniform_int_distribution<size_t> distributor(ai_func_type::FIRST,
                                                    ai_func_type::LAST);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    it->second.activation = (ai_func_type)get_rand(distributor, generator);
}

void pool::mutate_weight(genome &g) {
  exfloat step = mutation_rates.step_size;
  std::uniform_real_distribution<exfloat> real_distributor(0.0f, 1.0f);
  for (auto it = g.genes.begin(); it != g.genes.end(); it++) {
    if (get_rand(real_distributor, generator) < mutation_rates.perturb_chance)
      it->second.weight +=
          get_rand(real_distributor, generator) * step * 2.0f - step;
    else
      it->second.weight = get_rand(real_distributor, generator) * 4.0f - 2.0f;
  }
}

void pool::mutate_enable_disable(genome &g, const bool &enable) {
  std::vector<gene *> v;
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    if (it->second.enabled != enable)
      v.push_back(&it->second);
  if (v.size() == 0)
    return;
  std::uniform_int_distribution<int> distributor(0, v.size() - 1);
  v[get_rand(distributor, generator)]->enabled = enable;
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

void pool::mutate_link(genome &g, const bool &force_bias) {
  std::uniform_int_distribution<size_t> distributor1(0, g.max_neuron - 1);
  size_t neuron1 = get_rand(distributor1, generator);
  std::uniform_int_distribution<size_t> distributor2(
      network_info.input_size + network_info.bias_size, g.max_neuron - 1);
  size_t neuron2 = get_rand(distributor2, generator);
  if (is_output(neuron1, network_info) && is_output(neuron2, network_info))
    return;
  if (is_bias(neuron2, network_info))
    return;
  if (neuron1 == neuron2 && (!force_bias))
    return;
  if (is_output(neuron1, network_info))
    std::swap(neuron1, neuron2);
  if (force_bias) {
    std::uniform_int_distribution<size_t> bias_choose(
        network_info.input_size,
        network_info.input_size + network_info.output_size - 1);
    neuron1 = get_rand(bias_choose, generator);
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

      /*std::queue<size_t> que;
      std::vector<std::vector<size_t>> connections(g.max_neuron);
      for (auto it = g.genes.begin(); it != g.genes.end(); it++)
        connections[it->second.from_node].push_back(it->second.to_node);
      connections[neuron1].push_back(neuron2);
      for (size_t i = 0; i < connections[neuron1].size(); i++)
        que.push(connections[neuron1][i]);
      while (!que.empty()) {
        size_t tmp = que.front();
        if (tmp == neuron1) {
          has_recurrence = true;
          break;
        }
        que.pop();
        for (size_t i = 0; i < connections[tmp].size(); i++)
          que.push(connections[tmp][i]);
          }*/
    }

    if (has_recurrence)
      return;
  }

  gene new_gene;
  new_gene.from_node = neuron1;
  new_gene.to_node = neuron2;
  for (auto it = g.genes.begin(); it != g.genes.end(); it++)
    if (it->second.from_node == neuron1 && it->second.to_node == neuron2)
      return;

  {
    std::lock_guard<std::mutex> lock(inn_mutex);
    new_gene.innovation_num = innovation.add_gene(new_gene);
  }
  std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> act_generator(ai_func_type::FIRST,
                                                      ai_func_type::LAST);
  new_gene.weight = get_rand(weight_generator, generator) * 4.0f - 2.0f;
  new_gene.activation = (ai_func_type)get_rand(act_generator, generator);
  g.genes[new_gene.innovation_num] = new_gene;
}

void pool::mutate_neuron(genome &g) {
  if (g.genes.size() == 0)
    return;
  g.max_neuron++;
  std::uniform_int_distribution<size_t> distributor(0, g.genes.size() - 1);
  size_t gene_id = get_rand(distributor, generator);
  auto it = g.genes.begin();
  std::advance(it, gene_id);
  if (it->second.enabled == false)
    return;
  it->second.enabled = false;
  gene new_gene1;
  new_gene1.from_node = it->second.from_node;
  new_gene1.to_node = g.max_neuron - 1;
  std::uniform_real_distribution<exfloat> weight_generator(0.0f, 1.0f);
  new_gene1.weight = get_rand(weight_generator, generator);
  std::uniform_int_distribution<size_t> act_dist(ai_func_type::FIRST,
                                                 ai_func_type::LAST);
  new_gene1.activation = (ai_func_type)get_rand(act_dist, generator);
  {
    std::lock_guard<std::mutex> lock(inn_mutex);
    new_gene1.innovation_num = innovation.add_gene(new_gene1);
  }
  new_gene1.enabled = true;
  gene new_gene2;
  new_gene2.from_node = g.max_neuron - 1;
  new_gene2.to_node = it->second.to_node;
  new_gene2.weight = it->second.weight;
  new_gene2.activation = it->second.activation;
  {
    std::lock_guard<std::mutex> lock(inn_mutex);
    new_gene2.innovation_num = innovation.add_gene(new_gene2);
  }
  new_gene2.enabled = true;
  g.genes[new_gene1.innovation_num] = new_gene1;
  g.genes[new_gene2.innovation_num] = new_gene2;
}

void pool::mutate(genome &g) {
  exfloat coefficient[2] = {0.95f, 1.05263f};
  std::uniform_int_distribution<int> coin_flip(0, 1);
  g.mutation_rates.enable_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.disable_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.connection_mutate_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.neuron_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.link_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.bias_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.crossover_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.perturb_chance *=
      coefficient[get_rand(coin_flip, generator)];
  g.mutation_rates.activation_mutation_chance *=
      coefficient[get_rand(coin_flip, generator)];
  std::uniform_real_distribution<exfloat> mutate_or_not_mutate(0.0f, 1.0f);
  if (get_rand(mutate_or_not_mutate, generator) <
      g.mutation_rates.connection_mutate_chance)
    mutate_weight(g);

  if (get_rand(mutate_or_not_mutate, generator) <
      g.mutation_rates.activation_mutation_chance)
    mutate_activation(g);

  exfloat p = g.mutation_rates.link_mutation_chance;
  while (p > 0.0f) {
    if (get_rand(mutate_or_not_mutate, generator) < p)
      mutate_link(g, false);
    p -= 1.0f;
  }

  p = g.mutation_rates.neuron_mutation_chance;
  while (p > 0.0f) {
    if (get_rand(mutate_or_not_mutate, generator) < p)
      mutate_neuron(g);
    p -= 1.0f;
  }

  p = g.mutation_rates.bias_mutation_chance;
  while (p > 0.0f) {
    if (get_rand(mutate_or_not_mutate, generator) < p)
      mutate_link(g, true);
    p -= 1.0f;
  }

  p = g.mutation_rates.enable_mutation_chance;
  while (p > 0.0f) {
    if (get_rand(mutate_or_not_mutate, generator) < p)
      mutate_enable_disable(g, true);
    p -= 1.0f;
  }

  p = g.mutation_rates.disable_mutation_chance;
  while (p > 0.0f) {
    if (get_rand(mutate_or_not_mutate, generator) < p)
      mutate_enable_disable(g, false);
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

std::future<genome> pool::breed_child(specie &s) {
  genome child(network_info, mutation_rates);
  std::uniform_real_distribution<exfloat> distributor(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> choose_genome(0, s.genomes.size() - 1);
  if (get_rand(distributor, generator) < mutation_rates.crossover_chance) {
    genome &g1 = s.genomes[get_rand(choose_genome, generator)];
    genome &g2 = s.genomes[get_rand(choose_genome, generator)];
    child = crossover(g1, g2);
  } else {
    genome &g = s.genomes[get_rand(choose_genome, generator)];
    child = g;
  }

  return add_task(
      [this](genome child) {
        this->mutate(child);
        return child;
      },
      child);
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
  std::vector<std::future<bool>> f;
  auto s = species.begin();
  while (s != species.end()) {
    if (is_same_species(child, s->genomes[0])) {
      s->genomes.push_back(child);
      break;
    }
    s++;
  }

  if (s == species.end()) {
    specie new_specie;
    new_specie.genomes.push_back(child);
    species.push_back(new_specie);
  }
}

void pool::new_generation() {
  innovation.reset();
  cull_species(false);
  rank_globally();
  remove_stale_species();
  for (auto s = species.begin(); s != species.end(); s++)
    calculate_average_fitness(*s);
  remove_weak_species();
  std::vector<std::future<genome>> children;
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
          breed_child(*species_pointer[get_rand(choose_specie, generator)]));
  for (size_t i = 0; i < children.size(); i++)
    add_to_species(children[i].get());
  generation_number++;
}

std::istream &operator>>(std::istream &input, pool &p) {
  size_t innovation_num;
  input >> innovation_num;
  p.innovation.set_innovation_number(innovation_num);
  input >> p.generation_number;
  input >> p.max_fitness;
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
  output << p.innovation.number() << std::endl;
  output << p.generation_number << std::endl;
  output << p.max_fitness << std::endl;
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
