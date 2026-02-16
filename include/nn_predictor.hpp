/**
 * Neural Network Predictor for IBD Segment Classification
 *
 * Custom C++ implementation of a simple MLP (no external dependencies).
 * Loads weights from binary format exported by dev-notes/export_nn_weights.py.
 *
 * Architecture: 40 -> 36 -> 18 -> 9 -> 3 -> 1
 * Activations: LeakyReLU (hidden), Sigmoid (output)
 * Total parameters: 2,347
 */

#ifndef NN_PREDICTOR_HPP
#define NN_PREDICTOR_HPP

#include <string>
#include <vector>
#include <cstdint>

class NNPredictor {
public:
    /**
     * Constructor - loads weights from binary file
     * @param weights_path  Path to nn_weights.bin (exported by export_nn_weights.py)
     */
    explicit NNPredictor(const std::string& weights_path);

    ~NNPredictor() = default;

    // Non-copyable
    NNPredictor(const NNPredictor&) = delete;
    NNPredictor& operator=(const NNPredictor&) = delete;

    /**
     * Predict on batch of samples
     * @param features    Contiguous float array [n_samples x n_features], row-major
     * @param n_samples   Number of samples
     * @param n_features  Number of features per sample (must be 40)
     * @return            Vector of prediction probabilities (sigmoid output)
     */
    std::vector<float> predictBatch(const float* features, size_t n_samples, size_t n_features);

    /**
     * Check if weights loaded successfully
     */
    bool isLoaded() const { return loaded_; }

private:
    struct Layer {
        std::vector<float> weights;  // Row-major: [out_size x in_size]
        std::vector<float> biases;   // [out_size]
        int in_size;
        int out_size;
    };

    static constexpr uint32_t MAGIC = 0x4E4E5754;  // "NNWT"
    static constexpr uint32_t VERSION = 1;
    static constexpr float LEAKY_RELU_ALPHA = 0.01f;

    std::vector<Layer> layers_;
    bool loaded_ = false;

    bool loadWeights(const std::string& path);
    float forward(const float* input) const;
};

#endif // NN_PREDICTOR_HPP
