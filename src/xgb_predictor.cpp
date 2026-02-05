/**
 * XGBoost Predictor Implementation
 *
 * Wrapper around XGBoost C API for batch prediction of IBD segments.
 * Uses XGBPredictor class to load a trained model and classify segments
 * as true IBD (class 0) or false positive (class 1).
 */

#include "xgb_predictor.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

XGBPredictor::XGBPredictor(const std::string& model_path, int nthreads) {
    // Create booster
    int ret = XGBoosterCreate(nullptr, 0, &booster_);
    if (ret != 0) {
        throw std::runtime_error("Failed to create XGBoost booster: " + std::string(XGBGetLastError()));
    }

    // Set number of threads
    ret = XGBoosterSetParam(booster_, "nthread", std::to_string(nthreads).c_str());
    if (ret != 0) {
        XGBoosterFree(booster_);
        booster_ = nullptr;
        throw std::runtime_error("Failed to set nthread parameter: " + std::string(XGBGetLastError()));
    }

    // Load model
    ret = XGBoosterLoadModel(booster_, model_path.c_str());
    if (ret != 0) {
        XGBoosterFree(booster_);
        booster_ = nullptr;
        throw std::runtime_error("Failed to load XGBoost model from " + model_path + ": " + std::string(XGBGetLastError()));
    }
}

XGBPredictor::~XGBPredictor() {
    if (booster_ != nullptr) {
        XGBoosterFree(booster_);
    }
}

std::vector<float> XGBPredictor::predictBatch(const float* features, size_t n_samples, size_t n_features) {
    if (booster_ == nullptr) {
        throw std::runtime_error("XGBoost model not loaded");
    }

    if (n_samples == 0) {
        return {};
    }

    // Create DMatrix from feature array
    DMatrixHandle dmat;
    int ret = XGDMatrixCreateFromMat(features, n_samples, n_features, NAN, &dmat);
    if (ret != 0) {
        throw std::runtime_error("Failed to create DMatrix: " + std::string(XGBGetLastError()));
    }

    // Run prediction
    bst_ulong out_len = 0;
    const float* out_result = nullptr;

    // Use default prediction (probability for binary classification)
    ret = XGBoosterPredict(booster_, dmat, 0, 0, 0, &out_len, &out_result);
    if (ret != 0) {
        XGDMatrixFree(dmat);
        throw std::runtime_error("Prediction failed: " + std::string(XGBGetLastError()));
    }

    // Copy results
    std::vector<float> predictions(out_result, out_result + out_len);

    // Cleanup
    XGDMatrixFree(dmat);

    return predictions;
}
