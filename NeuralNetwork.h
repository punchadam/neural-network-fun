#pragma once

#include "layer.h"

template <typename T, typename LossFunction>
class NeuralNetwork {
    static_assert(std::is_floating_point<T>::value, "Type must be floating point");

private:
    // the entire network is just a vector of pointers to each layer
    std::vector<std::unique_ptr<Layer<T>>> layers;

public:
    NeuralNetwork() = default;

    void addLayer(std::unique_ptr<Layer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    void initializeAllWeights(std::mt19937& gen) {
        for (auto& layer : layers) {
            layer->initializeWeights(gen);
        }
    }

    Vector<T> forward(const Vector<T>& input) {
        Vector<T> activations = input;
        for (auto& layer : layers) {
            activations = layer->forwardPass(activations);
        }
        return activations;
    }

    T trainSample(const Vector<T>& input, const Vector<T>& target, T learningRate) {
        Vector<T> prediction = forward(input);
        LossFunction::calculateLoss(prediction, target);

        // loop backwards, calculating the gradient and passing to the previous layer
        Vector<T> gradient = LossFunction::calculateGradient(prediction, target);
        for (size_t i = layers.size() - 1; i-- > 0;) {
            gradient = layers[i]->backwardPass(gradient, learningRate);
        }
    }

};