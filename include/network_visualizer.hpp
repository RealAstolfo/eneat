#ifndef ENEAT_NETWORK_VISUALIZER_HPP
#define ENEAT_NETWORK_VISUALIZER_HPP

#include <vector>
#include "brain.hpp"
#include "raylib.h"

class NetworkVisualizer {
public:
    NetworkVisualizer(int width = 1400, int height = 1000);
    ~NetworkVisualizer();

    // Window management
    void open();
    void close();
    bool is_open() const;

    // Rendering
    void render(const brain& net);
    void update();

private:
    int width_;
    int height_;
    bool window_open_ = false;

    // Node position for layout
    struct NodePos {
        float x;
        float y;
    };

    // Compute topological depth for each neuron (BFS from inputs)
    std::vector<int> compute_depths(const brain& net);

    // Compute layered layout for the network (returns scale factor via out_scale)
    std::vector<NodePos> compute_layout(const brain& net, float& out_scale);

    // Color mapping functions
    Color value_to_color(float value);   // Neuron activation -> color
    Color weight_to_color(float weight); // Connection weight -> color
    Color type_to_border_color(neuron_place type); // Neuron type -> border color
};

#endif // ENEAT_NETWORK_VISUALIZER_HPP
