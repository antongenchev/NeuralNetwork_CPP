#pragma once

#include "layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    void addLayer(int input_size, int output_size);
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs, double learning_rate);
    std::vector<double> predict(const std::vector<double>& input);
private:
    std::vector<Layer> layers;
};