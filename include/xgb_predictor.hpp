/**
 * XGBoost Predictor for IBD Segment Classification
 *
 * Wrapper around XGBoost C API for batch prediction.
 */

#ifndef XGB_PREDICTOR_HPP
#define XGB_PREDICTOR_HPP

#include <string>
#include <vector>
#include <xgboost/c_api.h>

/**
 * XGBPredictor: Loads XGBoost model and performs batch prediction
 *
 * Usage:
 *   XGBPredictor predictor("model.json");
 *   std::vector<float> probs = predictor.predictBatch(features, n_samples, n_features);
 */
class XGBPredictor {
public:
    /**
     * Constructor - loads model from file
     * @param model_path  Path to XGBoost model (JSON or binary format)
     * @param nthreads    Number of threads for prediction (default: 1)
     */
    explicit XGBPredictor(const std::string& model_path, int nthreads = 1);

    ~XGBPredictor();

    // Non-copyable
    XGBPredictor(const XGBPredictor&) = delete;
    XGBPredictor& operator=(const XGBPredictor&) = delete;

    /**
     * Predict on batch of samples
     * @param features    Contiguous float array [n_samples x n_features], row-major
     * @param n_samples   Number of samples
     * @param n_features  Number of features per sample (should be 40)
     * @return            Vector of prediction probabilities (for positive class)
     */
    std::vector<float> predictBatch(const float* features, size_t n_samples, size_t n_features);

    /**
     * Check if model loaded successfully
     */
    bool isLoaded() const { return booster_ != nullptr; }

private:
    BoosterHandle booster_ = nullptr;
};

#endif // XGB_PREDICTOR_HPP
