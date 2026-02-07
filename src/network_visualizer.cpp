#include "network_visualizer.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include "rlgl.h"  // For rlEnableSmoothLines

NetworkVisualizer::NetworkVisualizer(int width, int height)
    : width_(width), height_(height), window_open_(false) {}

NetworkVisualizer::~NetworkVisualizer() {
    close();
}

void NetworkVisualizer::open() {
    if (!window_open_) {
        // Enable MSAA for smoother lines and circles, keep window running when unfocused
        SetConfigFlags(FLAG_WINDOW_ALWAYS_RUN | FLAG_MSAA_4X_HINT);
        InitWindow(width_, height_, "ENEAT Network Visualizer");
        SetWindowMinSize(width_, height_);
        SetWindowMaxSize(width_, height_);
        SetTargetFPS(60);

        // Enable OpenGL smooth lines for better anti-aliasing
        rlEnableSmoothLines();

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
    std::vector<bool> finalized(net.neurons.size(), false);

    // Build forward adjacency list (out_neurons) from in_connections
    std::vector<std::vector<size_t>> out_neurons(net.neurons.size());
    for (size_t i = 0; i < net.neurons.size(); i++) {
        for (const auto& conn : net.neurons[i].in_connections) {
            if (conn.from_neuron < net.neurons.size()) {
                out_neurons[conn.from_neuron].push_back(i);
            }
        }
    }

    // Bias neurons get special depth -2 (rendered separately at top)
    // Input neurons start at depth 0
    std::queue<size_t> q;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (net.neurons[i].type == BIAS) {
            depth[i] = -2;  // Special depth for bias neurons
            q.push(i);      // Still process their connections
        } else if (net.neurons[i].type == INPUT) {
            depth[i] = 0;
            q.push(i);
        }
    }

    // BFS with finalization to prevent infinite re-queuing in cycles
    // Once a node is finalized, it won't be re-processed even if reached again
    while (!q.empty()) {
        size_t current = q.front();
        q.pop();

        if (finalized[current]) continue;  // Already processed
        finalized[current] = true;

        // Bias neurons (depth -2) propagate as if at depth 0
        int current_effective_depth = (depth[current] == -2) ? 0 : depth[current];

        for (size_t next : out_neurons[current]) {
            if (next == current || finalized[next]) continue;  // Skip self-loops and finalized nodes

            int new_depth = current_effective_depth + 1;
            if (new_depth > depth[next]) {
                depth[next] = new_depth;
                q.push(next);
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

    // Fallback: any hidden neuron still at depth -1 goes to layer 1
    // This handles neurons only reachable via recurrent connections
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (net.neurons[i].type == HIDDEN && depth[i] == -1) {
            depth[i] = 1;
        }
    }

    return depth;
}

std::vector<NetworkVisualizer::NodePos> NetworkVisualizer::compute_layout(const brain& net, float& out_scale) {
    std::vector<NodePos> positions(net.neurons.size());

    // Use actual screen dimensions
    const float screen_width = static_cast<float>(GetScreenWidth());
    const float screen_height = static_cast<float>(GetScreenHeight());

    // Layout constants (same margins as before)
    const float BASE_CELL_SIZE = 20.0f;   // Base grid cell size for hidden layers
    const float MARGIN = 60.0f;           // Screen margin
    const float BOTTOM_RESERVE = 50.0f;   // Reserve for info text
    const float BIAS_ROW_HEIGHT = 30.0f;  // Space for horizontal bias row at top

    // Compute topological depth for each neuron
    auto depths = compute_depths(net);

    // Collect bias neurons (depth = -2) separately
    std::vector<size_t> bias_neurons;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (depths[i] == -2) {
            bias_neurons.push_back(i);
        }
    }

    // Fixed Y bounds for main network (below bias row)
    const float bias_y = MARGIN;  // Bias neurons at top
    const float top_y = MARGIN + BIAS_ROW_HEIGHT;  // Main network starts below bias
    const float bottom_y = screen_height - MARGIN - BOTTOM_RESERVE;
    const float available_height = bottom_y - top_y;
    const float available_width = screen_width - 2 * MARGIN;

    // Find max depth (number of layers - 1), excluding bias neurons
    int max_depth = 0;
    for (int d : depths) {
        if (d > max_depth) max_depth = d;
    }
    max_depth = std::min(max_depth, 100);  // Safety clamp
    if (max_depth < 1) max_depth = 1;

    // Group neurons by their depth (layer), excluding bias (depth -2)
    std::vector<std::vector<size_t>> layers(max_depth + 1);
    for (size_t i = 0; i < net.neurons.size(); i++) {
        int d = depths[i];
        if (d >= 0 && d <= max_depth) {
            layers[d].push_back(i);
        }
    }

    // Find max neurons in any hidden layer (for grid scaling)
    size_t max_hidden_layer_size = 1;
    for (int layer_idx = 1; layer_idx < max_depth; layer_idx++) {
        if (layers[layer_idx].size() > max_hidden_layer_size) {
            max_hidden_layer_size = layers[layer_idx].size();
        }
    }

    // Calculate scale: ensure all neurons in tallest hidden layer fit
    // Each neuron needs BASE_CELL_SIZE vertical space
    float needed_height = max_hidden_layer_size * BASE_CELL_SIZE;
    out_scale = std::min(1.0f, available_height / needed_height);

    // Actual cell size after scaling
    float cell_size = BASE_CELL_SIZE * out_scale;

    // Count hidden layers for X positioning
    int hidden_layer_count = 0;
    for (int layer_idx = 1; layer_idx < max_depth; layer_idx++) {
        if (!layers[layer_idx].empty()) {
            hidden_layer_count++;
        }
    }

    // Position INPUT layer (layer 0) - spans full height like before
    if (!layers[0].empty()) {
        float x = MARGIN;
        size_t layer_size = layers[0].size();
        if (layer_size == 1) {
            positions[layers[0][0]] = {x, (top_y + bottom_y) / 2.0f};
        } else {
            float spacing = available_height / (layer_size - 1);
            for (size_t i = 0; i < layer_size; i++) {
                float y = top_y + i * spacing;
                positions[layers[0][i]] = {x, y};
            }
        }
    }

    // Position OUTPUT layer (layer max_depth) - spans full height like before
    if (!layers[max_depth].empty()) {
        float x = MARGIN + available_width;
        size_t layer_size = layers[max_depth].size();
        if (layer_size == 1) {
            positions[layers[max_depth][0]] = {x, (top_y + bottom_y) / 2.0f};
        } else {
            float spacing = available_height / (layer_size - 1);
            for (size_t i = 0; i < layer_size; i++) {
                float y = top_y + i * spacing;
                positions[layers[max_depth][i]] = {x, y};
            }
        }
    }

    // Position HIDDEN layers on a grid (centered vertically)
    if (hidden_layer_count > 0) {
        // X spacing: distribute hidden layers between input and output
        float hidden_start_x = MARGIN + available_width * 0.1f;  // 10% from input
        float hidden_end_x = MARGIN + available_width * 0.9f;    // 10% from output
        float hidden_width = hidden_end_x - hidden_start_x;

        int hidden_idx = 0;
        for (int layer_idx = 1; layer_idx < max_depth; layer_idx++) {
            const auto& layer = layers[layer_idx];
            if (layer.empty()) continue;

            // X position for this hidden layer
            float x = (hidden_layer_count == 1)
                ? (hidden_start_x + hidden_end_x) / 2.0f
                : hidden_start_x + (static_cast<float>(hidden_idx) / (hidden_layer_count - 1)) * hidden_width;

            // Y positions: grid cells centered vertically
            size_t layer_size = layer.size();
            float grid_height = layer_size * cell_size;
            float grid_start_y = top_y + (available_height - grid_height) / 2.0f;

            for (size_t i = 0; i < layer_size; i++) {
                float y = grid_start_y + i * cell_size + cell_size / 2.0f;
                positions[layer[i]] = {x, y};
            }

            hidden_idx++;
        }
    }

    // Position bias neurons horizontally at the top, aligned with hidden layers
    if (!bias_neurons.empty()) {
        size_t bias_count = bias_neurons.size();

        if (hidden_layer_count > 0 && bias_count <= static_cast<size_t>(hidden_layer_count)) {
            // Align bias neurons with hidden layer columns
            std::vector<float> hidden_x_positions;
            float hidden_start_x = MARGIN + available_width * 0.1f;
            float hidden_end_x = MARGIN + available_width * 0.9f;
            float hidden_width = hidden_end_x - hidden_start_x;

            int hidden_idx = 0;
            for (int layer_idx = 1; layer_idx < max_depth; layer_idx++) {
                if (!layers[layer_idx].empty()) {
                    float x = (hidden_layer_count == 1)
                        ? (hidden_start_x + hidden_end_x) / 2.0f
                        : hidden_start_x + (static_cast<float>(hidden_idx) / (hidden_layer_count - 1)) * hidden_width;
                    hidden_x_positions.push_back(x);
                    hidden_idx++;
                }
            }

            for (size_t i = 0; i < bias_count; i++) {
                size_t x_idx = (hidden_x_positions.size() > 1)
                    ? (i * (hidden_x_positions.size() - 1)) / (bias_count > 1 ? bias_count - 1 : 1)
                    : 0;
                x_idx = std::min(x_idx, hidden_x_positions.size() - 1);
                positions[bias_neurons[i]] = {hidden_x_positions[x_idx], bias_y};
            }
        } else {
            // No hidden layers or more bias than hidden: spread evenly
            float bias_left = MARGIN + available_width * 0.15f;
            float bias_right = MARGIN + available_width * 0.85f;
            float bias_width = bias_right - bias_left;

            if (bias_count == 1) {
                positions[bias_neurons[0]] = {(bias_left + bias_right) / 2.0f, bias_y};
            } else {
                for (size_t i = 0; i < bias_count; i++) {
                    float x = bias_left + (static_cast<float>(i) / (bias_count - 1)) * bias_width;
                    positions[bias_neurons[i]] = {x, bias_y};
                }
            }
        }
    }

    // Handle disconnected neurons (depth = -1) - place them at far right
    std::vector<size_t> disconnected;
    for (size_t i = 0; i < net.neurons.size(); i++) {
        if (depths[i] == -1) {
            disconnected.push_back(i);
        }
    }
    if (!disconnected.empty()) {
        float x = MARGIN + available_width + 30;
        size_t disc_size = disconnected.size();
        if (disc_size == 1) {
            positions[disconnected[0]] = {x, (top_y + bottom_y) / 2.0f};
        } else {
            float spacing = available_height / (disc_size - 1);
            for (size_t i = 0; i < disc_size; i++) {
                float y = top_y + i * spacing;
                positions[disconnected[i]] = {x, y};
            }
        }
    }

    return positions;
}

void NetworkVisualizer::render_network_only(const brain& net) {
    if (!window_open_) return;

    // Force window to stay at fixed size
    if (GetScreenWidth() != width_ || GetScreenHeight() != height_) {
        SetWindowSize(width_, height_);
    }

    float scale = 1.0f;
    auto positions = compute_layout(net, scale);

    // Scale neuron radius based on zoom level (minimum 2px for visibility)
    const float base_radius = 6.0f;
    const float neuron_radius = std::max(base_radius * scale, 2.0f);

    // Count connections for info display
    size_t connection_count = 0;
    size_t hebbian_count = 0;
    for (const auto& n : net.neurons) {
        connection_count += n.in_connections.size();
        for (const auto& conn : n.in_connections) {
            if (conn.trait_id > 0 && conn.trait_id <= net.traits.size()) {
                if (net.traits[conn.trait_id - 1].is_learning_enabled()) {
                    hebbian_count++;
                }
            }
        }
    }

    // Draw connections first (so they're behind neurons)
    for (size_t to_idx = 0; to_idx < net.neurons.size(); to_idx++) {
        const auto& neuron = net.neurons[to_idx];
        const auto& to_pos = positions[to_idx];

        for (const auto& conn : neuron.in_connections) {
            if (conn.from_neuron < positions.size()) {
                const auto& from_pos = positions[conn.from_neuron];

                Color line_color = weight_to_color(conn.weight);

                // Highlight Hebbian connections with a different style
                bool is_hebbian = false;
                if (conn.trait_id > 0 && conn.trait_id <= net.traits.size()) {
                    is_hebbian = net.traits[conn.trait_id - 1].is_learning_enabled();
                }

                if (is_hebbian) {
                    // Hebbian connections get a purple tint
                    line_color.r = (line_color.r + 128) / 2;
                    line_color.b = (line_color.b + 255) / 2;
                }

                // Minimum 1.0 thickness for better visibility with MSAA
                float thickness = std::clamp(std::abs(conn.weight) * 2.0f * scale, 1.0f, 4.0f);

                DrawLineEx(
                    Vector2{from_pos.x, from_pos.y},
                    Vector2{to_pos.x, to_pos.y},
                    thickness,
                    line_color
                );

                // Draw recurrent indicator (small arrow) if recurrent
                if (conn.is_recurrent) {
                    float mid_x = (from_pos.x + to_pos.x) / 2.0f;
                    float mid_y = (from_pos.y + to_pos.y) / 2.0f;
                    DrawCircle(static_cast<int>(mid_x), static_cast<int>(mid_y), 3.0f, PURPLE);
                }
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
    if (hebbian_count > 0) {
        DrawText(
            TextFormat("Neurons: %d  |  Connections: %d  |  Hebbian: %d  |  Scale: %.0f%%",
                       (int)net.neurons.size(), (int)connection_count, (int)hebbian_count, scale * 100),
            10, screen_h - 25, 16, LIGHTGRAY
        );
    } else {
        DrawText(
            TextFormat("Neurons: %d  |  Connections: %d  |  Scale: %.0f%%",
                       (int)net.neurons.size(), (int)connection_count, scale * 100),
            10, screen_h - 25, 16, LIGHTGRAY
        );
    }

    // Draw legend
    DrawText("Input", 10, 10, 12, GREEN);
    DrawText("Output", 60, 10, 12, ORANGE);
    DrawText("Hidden", 120, 10, 12, GRAY);
    DrawText("Bias", 180, 10, 12, YELLOW);
    if (net.hebbian_enabled) {
        DrawText("Hebbian", 230, 10, 12, PURPLE);
    }

    // Call label callback if set (allows custom labels to be drawn)
    if (label_callback_) {
        for (size_t i = 0; i < net.neurons.size(); i++) {
            label_callback_(i, positions[i].x, positions[i].y, net.neurons[i]);
        }
    }
}

void NetworkVisualizer::render(const brain& net) {
    if (!window_open_) return;

    // Force window to stay at fixed size
    if (GetScreenWidth() != width_ || GetScreenHeight() != height_) {
        SetWindowSize(width_, height_);
    }

    BeginDrawing();
    ClearBackground(Color{30, 30, 30, 255}); // Dark background

    render_network_only(net);

    EndDrawing();
}

void NetworkVisualizer::set_label_callback(NeuronLabelCallback callback) {
    label_callback_ = std::move(callback);
}
