/**
 * Neural Network Predictor Implementation
 *
 * Implements a 5-layer MLP for IBD segment false-positive classification,
 * matching the PyTorch reference model in segmentAugmentation.py.
 *
 * Binary weight format (produced by dev-notes/export_nn_weights.py):
 *   Header:  magic (0x4E4E5754 "NNWT"), version (uint32), n_layers (uint32)
 *   Per layer: in_size (uint32), out_size (uint32),
 *              weights (float32 × out_size × in_size, row-major),
 *              biases  (float32 × out_size)
 *
 * Validated against Python reference: max prediction diff < 6e-4 (float32).
 */

#include "nn_predictor.hpp"
#include <cstdio>
#include <cmath>
#include <iostream>

NNPredictor::NNPredictor(const std::string& weights_path) {
    loaded_ = loadWeights(weights_path);
}

/**
 * Load network weights from binary file.
 * File format: header (magic, version, n_layers) + per-layer (dims, weights, biases).
 */
bool NNPredictor::loadWeights(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "[NNPredictor] Cannot open weights file: " << path << "\n";
        return false;
    }

    // Read and validate header
    uint32_t magic, version, n_layers;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&n_layers, sizeof(uint32_t), 1, f) != 1) {
        std::cerr << "[NNPredictor] Failed to read header\n";
        fclose(f);
        return false;
    }

    if (magic != MAGIC) {
        std::cerr << "[NNPredictor] Invalid magic number: 0x" << std::hex << magic
                  << " (expected 0x" << MAGIC << ")\n" << std::dec;
        fclose(f);
        return false;
    }
    if (version != VERSION) {
        std::cerr << "[NNPredictor] Unsupported version: " << version << "\n";
        fclose(f);
        return false;
    }

    // Read each layer's dimensions, weights, and biases
    layers_.resize(n_layers);
    for (uint32_t i = 0; i < n_layers; ++i) {
        uint32_t in_size, out_size;
        if (fread(&in_size, sizeof(uint32_t), 1, f) != 1 ||
            fread(&out_size, sizeof(uint32_t), 1, f) != 1) {
            std::cerr << "[NNPredictor] Failed to read layer " << i << " dimensions\n";
            fclose(f);
            return false;
        }

        Layer& layer = layers_[i];
        layer.in_size = static_cast<int>(in_size);
        layer.out_size = static_cast<int>(out_size);

        // Weights stored row-major: [out_size × in_size]
        size_t weight_count = static_cast<size_t>(in_size) * out_size;
        layer.weights.resize(weight_count);
        if (fread(layer.weights.data(), sizeof(float), weight_count, f) != weight_count) {
            std::cerr << "[NNPredictor] Failed to read layer " << i << " weights\n";
            fclose(f);
            return false;
        }

        // Biases: [out_size]
        layer.biases.resize(out_size);
        if (fread(layer.biases.data(), sizeof(float), out_size, f) != out_size) {
            std::cerr << "[NNPredictor] Failed to read layer " << i << " biases\n";
            fclose(f);
            return false;
        }
    }

    fclose(f);
    return true;
}

/**
 * Forward pass for a single sample through the 5-layer MLP.
 *
 * Architecture matches PyTorch reference:
 *   Layer 1: Linear(40→36)  + LeakyReLU(α=0.01)
 *   Layer 2: Linear(36→18)  + LeakyReLU(α=0.01)
 *   Layer 3: Linear(18→9)   + LeakyReLU(α=0.01)
 *   Layer 4: Linear(9→3)    + LeakyReLU(α=0.01)
 *   Output:  Linear(3→1)    + Sigmoid
 *
 * Uses fixed-size stack arrays for activations (no heap allocation).
 * Input must be StandardScaler-normalized (same normalization as XGBoost path).
 *
 * @param input  Pointer to 40 normalized float features
 * @return       Probability of segment being a false positive (≥0.5 → filter)
 */
float NNPredictor::forward(const float* input) const {
    float h1[36], h2[18], h3[9], h4[3];

    // Layer 1: 40 → 36, LeakyReLU
    const Layer& l1 = layers_[0];
    for (int i = 0; i < 36; ++i) {
        float sum = l1.biases[i];
        const float* w = l1.weights.data() + i * 40;
        for (int j = 0; j < 40; ++j)
            sum += w[j] * input[j];
        h1[i] = sum > 0.0f ? sum : LEAKY_RELU_ALPHA * sum;
    }

    // Layer 2: 36 → 18, LeakyReLU
    const Layer& l2 = layers_[1];
    for (int i = 0; i < 18; ++i) {
        float sum = l2.biases[i];
        const float* w = l2.weights.data() + i * 36;
        for (int j = 0; j < 36; ++j)
            sum += w[j] * h1[j];
        h2[i] = sum > 0.0f ? sum : LEAKY_RELU_ALPHA * sum;
    }

    // Layer 3: 18 → 9, LeakyReLU
    const Layer& l3 = layers_[2];
    for (int i = 0; i < 9; ++i) {
        float sum = l3.biases[i];
        const float* w = l3.weights.data() + i * 18;
        for (int j = 0; j < 18; ++j)
            sum += w[j] * h2[j];
        h3[i] = sum > 0.0f ? sum : LEAKY_RELU_ALPHA * sum;
    }

    // Layer 4: 9 → 3, LeakyReLU
    const Layer& l4 = layers_[3];
    for (int i = 0; i < 3; ++i) {
        float sum = l4.biases[i];
        const float* w = l4.weights.data() + i * 9;
        for (int j = 0; j < 9; ++j)
            sum += w[j] * h3[j];
        h4[i] = sum > 0.0f ? sum : LEAKY_RELU_ALPHA * sum;
    }

    // Output: 3 → 1, Sigmoid → P(false positive)
    const Layer& l5 = layers_[4];
    float out = l5.biases[0];
    for (int j = 0; j < 3; ++j)
        out += l5.weights[j] * h4[j];

    return 1.0f / (1.0f + std::exp(-out));
}

/**
 * Batch prediction using OpenMP parallelism.
 * Each sample's forward pass is independent, so parallelizes trivially.
 *
 * @param features    Row-major float matrix [n_samples × n_features]
 * @param n_samples   Number of segments to classify
 * @param n_features  Features per segment (must be 40)
 * @return            Vector of P(false positive) values, one per segment
 */
std::vector<float> NNPredictor::predictBatch(const float* features, size_t n_samples, size_t n_features) {
    std::vector<float> predictions(n_samples);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_samples; ++i) {
        predictions[i] = forward(features + i * n_features);
    }

    return predictions;
}
