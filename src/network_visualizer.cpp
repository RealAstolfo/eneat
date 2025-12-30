#include "network_visualizer.hpp"
#include <algorithm>
#include <cmath>
#include <queue>

NetworkVisualizer::NetworkVisualizer(int width, int height)
    : width_(width), height_(height), window_open_(false) {}

NetworkVisualizer::~NetworkVisualizer() {
    close();
}

void NetworkVisualizer::open() {
    if (!window_open_) {
        // Prevent window resizing
        SetConfigFlags(FLAG_WINDOW_ALWAYS_RUN);
        InitWindow(width_, height_, "ENEAT Network Visualizer");
        SetWindowMinSize(width_, height_);
        SetWindowMaxSize(width_, height_);
        SetTargetFPS(60);
        window_open_ = true;
    }
}

void NetworkVisualizer::close() {
    if (window_open_) {
        CloseWindow();
        window_open_ = false;
    }
}

bool NetworkVisualizer::is_open() const {
    return window_open_ && !WindowShouldClose();
}

void NetworkVisualizer::update() {
    if (window_open_ && WindowShouldClose()) {
        close();
    }
}

Color NetworkVisualizer::value_to_color(float v) {
    // Clamp value to [-1, 1] range using tanh-like normalization
    v = std::tanh(v);

    if (v < 0) {
        // Negative: blue gradient (more negative = more saturated blue)
        unsigned char intensity = static_cast<unsigned char>(255 * std::abs(v));
        return Color{50, 50, static_cast<unsigned char>(100 + intensity * 0.6f), 255};
    } else {
        // Positive: red gradient (more positive = more saturated red)
        unsigned char intensity = static_cast<unsigned char>(255 * v);
        return Color{static_cast<unsigned char>(100 + intensity * 0.6f), 50, 50, 255};
    }
}

Color NetworkVisualizer::weight_to_color(float w) {
    // Clamp weight visualization
    w = std::clamp(w, -2.0f, 2.0f) / 2.0f;

    if (w < 0) {
        // Negative weights: blue
        unsigned char intensity = static_cast<unsigned char>(255 * std::abs(w));
        return Color{0, 0, intensity, 200};
    } else {
        // Positive weights: red
        unsigned char intensity = static_cast<unsigned char>(255 * w);
        return Color{intensity, 0, 0, 200};
    }
}

Color NetworkVisualizer::type_to_border_color(neuron_place type) {
    switch (type) {
        case INPUT:  return GREEN;
        case OUTPUT: return ORANGE;
        case BIAS:   return YELLOW;
        case HIDDEN:
        default:     return GRAY;
    }
}

std::vector<int> NetworkVisualizer::compute_depths(const brain& net) {
    std::vector<int> depth(net.neurons.size(), -1);

    // Build forward adjacency list (out_neurons) from in_neurons
    std::vector<std::vector<size_t>> out_neurons(net.neurons.size());
    for (size_t i = 0; i < net.neurons.size(); i++) {
        for (const auto& [from, weight] : net.neurons[i].in_neurons) {
            if (from < net.neurons.size()) {
                out_neurons[from].push_back(i);
            }
        }
    }

    // Initialize input/bias neurons at depth 0
    std::queue<size_t> q;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (net.neurons[i].type == INPUT || net.neurons[i].type == BIAS) {
            depth[i] = 0;
            q.push(i);
        }
    }

    // BFS: propagate depth through connections
    // For each neuron, depth = max(depth of all inputs) + 1
    // Cap depth to prevent runaway in cyclic/recurrent networks
    constexpr int MAX_DEPTH = 100;

    while (!q.empty()) {
        size_t current = q.front();
        q.pop();

        for (size_t next : out_neurons[current]) {
            int new_depth = depth[current] + 1;
            if (new_depth > depth[next] && new_depth <= MAX_DEPTH) {
                depth[next] = new_depth;
                q.push(next);  // Re-process to propagate max depth
            }
        }
    }

    // Find max depth among non-output neurons (hidden layers)
    int max_hidden_depth = 0;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (net.neurons[i].type != OUTPUT && depth[i] > max_hidden_depth) {
            max_hidden_depth = depth[i];
        }
    }

    // Normalize: all OUTPUT neurons go to rightmost layer (max_hidden_depth + 1)
    // This ensures outputs are always at the same X position regardless of connectivity
    int output_depth = max_hidden_depth + 1;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (net.neurons[i].type == OUTPUT) {
            depth[i] = output_depth;
        }
    }

    return depth;
}

std::vector<NetworkVisualizer::NodePos> NetworkVisualizer::compute_layout(const brain& net, float& out_scale) {
    std::vector<NodePos> positions(net.neurons.size());

    // Use actual screen dimensions
    const float screen_width = static_cast<float>(GetScreenWidth());
    const float screen_height = static_cast<float>(GetScreenHeight());

    // Layout constants
    const float MIN_SPACING = 8.0f;   // Reduced for large networks
    const float NEURON_RADIUS = 6.0f; // Smaller neurons for dense layers
    const float MARGIN = 40.0f;

    const float available_height = screen_height - 2 * MARGIN - 30; // Reserve for info text
    const float available_width = screen_width - 2 * MARGIN;
    const float center_y = MARGIN + available_height / 2.0f;  // Center of available area, not screen

    // Compute topological depth for each neuron
    auto depths = compute_depths(net);

    // Find max depth (number of layers - 1)
    int max_depth = 0;
    for (int d : depths) {
        if (d > max_depth) max_depth = d;
    }
    // Safety clamp to prevent huge allocations
    max_depth = std::min(max_depth, 100);
    if (max_depth < 1) max_depth = 1;

    // Group neurons by their depth (layer)
    std::vector<std::vector<size_t>> layers(max_depth + 1);
    for (size_t i = 0; i < net.neurons.size(); i++) {
        int d = depths[i];
        if (d >= 0 && d <= max_depth) {
            layers[d].push_back(i);
        }
    }

    // Find the tallest layer for scaling
    size_t max_layer_size = 1;
    for (const auto& layer : layers) {
        if (layer.size() > max_layer_size) {
            max_layer_size = layer.size();
        }
    }

    // Calculate scale to fit tallest layer
    // Span of n neurons = (n-1) * spacing between centers
    // We need this span to fit within available_height
    float base_spacing = 2 * NEURON_RADIUS + MIN_SPACING;
    float required_height = (max_layer_size - 1) * base_spacing;
    float scale_y = (required_height > 0) ? (available_height / required_height) : 1.0f;
    out_scale = std::min(scale_y, 1.0f);

    // Position neurons by layer
    // X: evenly distribute layers across available width
    // Y: center neurons within each layer
    float neuron_spacing = (2 * NEURON_RADIUS + MIN_SPACING) * out_scale;

    for (int layer_idx = 0; layer_idx <= max_depth; layer_idx++) {
        const auto& layer = layers[layer_idx];
        if (layer.empty()) continue;

        // X position: linear interpolation from left margin to right margin
        float x = MARGIN + (static_cast<float>(layer_idx) / max_depth) * available_width;

        // Y positions: center the layer vertically
        // Total span from first to last neuron center = (n-1) * spacing
        size_t layer_size = layer.size();
        float total_span = (layer_size - 1) * neuron_spacing;
        float start_y = center_y - total_span / 2.0f;

        for (size_t i = 0; i < layer_size; i++) {
            float y = start_y + i * neuron_spacing;
            positions[layer[i]] = {x, y};
        }
    }

    // Handle disconnected neurons (depth = -1) - place them at far right
    std::vector<size_t> disconnected;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (depths[i] < 0) {
            disconnected.push_back(i);
        }
    }
    if (!disconnected.empty()) {
        float x = MARGIN + available_width + 30; // Past the output layer
        float total_span = (disconnected.size() - 1) * neuron_spacing;
        float start_y = center_y - total_span / 2.0f;

        for (size_t i = 0; i < disconnected.size(); i++) {
            float y = start_y + i * neuron_spacing;
            positions[disconnected[i]] = {x, y};
        }
    }

    return positions;
}

void NetworkVisualizer::render(const brain& net) {
    if (!window_open_) return;

    // Force window to stay at fixed size
    if (GetScreenWidth() != width_ || GetScreenHeight() != height_) {
        SetWindowSize(width_, height_);
    }

    BeginDrawing();
    ClearBackground(Color{30, 30, 30, 255}); // Dark background

    float scale = 1.0f;
    auto positions = compute_layout(net, scale);

    // Scale neuron radius based on zoom level (minimum 2px for visibility)
    const float base_radius = 6.0f;
    const float neuron_radius = std::max(base_radius * scale, 2.0f);

    // Count connections for info display
    size_t connection_count = 0;
    for (const auto& n : net.neurons) {
        connection_count += n.in_neurons.size();
    }

    // Draw connections first (so they're behind neurons)
    for (size_t to_idx = 0; to_idx < net.neurons.size(); to_idx++) {
        const auto& neuron = net.neurons[to_idx];
        const auto& to_pos = positions[to_idx];

        for (const auto& [from_idx, weight] : neuron.in_neurons) {
            if (from_idx < positions.size()) {
                const auto& from_pos = positions[from_idx];

                Color line_color = weight_to_color(weight);
                float thickness = std::clamp(std::abs(weight) * 2.0f * scale, 0.5f, 3.0f);

                DrawLineEx(
                    Vector2{from_pos.x, from_pos.y},
                    Vector2{to_pos.x, to_pos.y},
                    thickness,
                    line_color
                );
            }
        }
    }

    // Draw neurons
    for (size_t i = 0; i < net.neurons.size(); i++) {
        const auto& neuron = net.neurons[i];
        const auto& pos = positions[i];

        // Fill color based on value
        Color fill = value_to_color(neuron.value);
        DrawCircle(static_cast<int>(pos.x), static_cast<int>(pos.y), neuron_radius, fill);

        // Border based on type
        Color border = type_to_border_color(neuron.type);
        DrawCircleLines(static_cast<int>(pos.x), static_cast<int>(pos.y), neuron_radius, border);
        DrawCircleLines(static_cast<int>(pos.x), static_cast<int>(pos.y), neuron_radius + 1, border);
    }

    // Draw info text (use actual screen height)
    int screen_h = GetScreenHeight();
    DrawText(
        TextFormat("Neurons: %d  |  Connections: %d  |  Scale: %.0f%%",
                   (int)net.neurons.size(), (int)connection_count, scale * 100),
        10, screen_h - 25, 16, LIGHTGRAY
    );

    // Draw legend
    DrawText("Input", 10, 10, 12, GREEN);
    DrawText("Output", 60, 10, 12, ORANGE);
    DrawText("Hidden", 120, 10, 12, GRAY);
    DrawText("Bias", 180, 10, 12, YELLOW);

    EndDrawing();
}
