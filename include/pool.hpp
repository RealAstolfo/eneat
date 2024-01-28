#ifndef ENEAT_POOL_HPP
#define ENEAT_POOL_HPP

#include <cstddef>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <random>
#include <stdbool.h>
#include <utility>
#include <vector>

#include "innovation_container.hpp"
#include "mutation_rate_container.hpp"
#include "network_info_container.hpp"
#include "speciating_parameter_container.hpp"

typedef struct specie specie;
typedef struct genome genome;

struct pool {
  innovation_container innovation;
  std::mutex inn_mutex;

  std::map<std::pair<size_t, size_t>, size_t> track;
  size_t generation_number = 1;
  size_t max_fitness = 0;
  mutation_rate_container mutation_rates;
  speciating_parameter_container speciating_parameters;
  network_info_container network_info;
  std::list<specie> species;

  std::random_device rd;
  std::mt19937 generator;
  std::mutex gen_mutex;

  pool(size_t input, size_t output, size_t bias = 1, bool rec = false);

  bool is_same_species(const genome &g1, const genome &g2);
  void new_generation();
  std::vector<std::pair<specie *, genome *>> get_genomes();
  genome crossover(const genome &g1, const genome &g2);
  void mutate_activation(genome &g);
  void mutate_weight(genome &g);
  void mutate_enable_disable(genome &g, const bool &enable);
  void mutate_link(genome &g, const bool &force_bias);
  void mutate_neuron(genome &g);
  void mutate(genome &g);
  exfloat disjoint(const genome &g1, const genome &g2);
  exfloat weights(const genome &g1, const genome &g2);
  void rank_globally();
  void calculate_average_fitness(specie &s);
  size_t total_average_fitness();
  void cull_species(const bool &cut_to_one);
  std::future<genome> breed_child(specie &s);
  void remove_stale_species();
  void remove_weak_species();
  void add_to_species(const genome &child);

  inline exfloat get_rand(std::uniform_real_distribution<exfloat> &d,
                          std::mt19937 &generator) {
    std::lock_guard<std::mutex> lock(gen_mutex);
    return d(generator);
  }

  inline size_t get_rand(std::uniform_int_distribution<size_t> &d,
                         std::mt19937 &generator) {
    std::lock_guard<std::mutex> lock(gen_mutex);
    return d(generator);
  }

  inline int get_rand(std::uniform_int_distribution<int> &d,
                      std::mt19937 &generator) {
    std::lock_guard<std::mutex> lock(gen_mutex);
    return d(generator);
  }
};

std::istream &operator>>(std::istream &input, pool &p);
std::ostream &operator<<(std::ostream &output, pool &p);

#endif
