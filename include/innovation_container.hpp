#ifndef INNOVATION_CONTAINER_HPP
#define INNOVATION_CONTAINER_HPP

#include <map>
#include <stdlib.h>
#include <utility>

#include "gene.hpp"

struct innovation_container {
  size_t _number;
  std::map<std::pair<size_t, size_t>, size_t> track;

  innovation_container() : _number(0) {}
  inline void set_innovation_number(size_t num);
  inline size_t add_gene(gene &g);
  inline void reset();
  size_t number() { return _number; }
};

void innovation_container::set_innovation_number(size_t num) {
  _number = num;
  reset();
}

size_t innovation_container::add_gene(gene &g) {
  auto it = track.find(std::make_pair(g.from_node, g.to_node));
  if (it == track.end())
    return track[std::make_pair(g.from_node, g.to_node)] = ++_number;
  else
    return it->second;
}

void innovation_container::reset() { track.clear(); }

#endif
