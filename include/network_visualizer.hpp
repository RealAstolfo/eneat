#ifndef ENEAT_NETWORK_VISUALIZER_HPP
#define ENEAT_NETWORK_VISUALIZER_HPP

#include <vector>
#include <functional>
#include "brain.hpp"
#include "raylib.h"

// Callback receives: neuron index, x position, y position, neuron value
using NeuronLabelCallback = std::function<void(size_t idx, float x, float y, float value)>;

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

    // Optional callback for drawing labels on neurons
    void set_label_callback(NeuronLabelCallback callback);

private:
    int width_;
    int height_;
    bool window_open_ = false;
    NeuronLabelCallback label_callback_;

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
