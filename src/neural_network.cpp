#include "neural_network.h"
#include "loss.h"
#include <iostream>


void NeuralNetwork::addLayer(int input_size, int output_size) {
    // Create a new layer and add it to the layers vector
    layers.emplace_back(input_size, output_size);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    // Propagate through all layers
    std::vector<double> current_output = input;
    for (auto& layer : layers) {
        current_output = layer.forward(current_output);
    }
    return current_output;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs, double learning_rate) {
    if (X.size() != y.size()) {
        std::cerr << "Error: Input data and labels size mismatch" << std::endl;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        // Iterate over each training example
        for (size_t i = 0; i < X.size(); ++i) {
            // Forward pass
            std::vector<double> output = predict(X[i]);
            // Calculate loss
            double loss = mse(output, y[i]);
            total_loss += loss;
            // Backward pass
            std::vector<double> loss_gradient = mse_derivative(output, y[i]);

            // Propagate the gradient backward
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                loss_gradient = it->backward(loss_gradient, learning_rate);
            }
        }

        // Print average loss for the epoch
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << (total_loss / X.size()) << std::endl;
    }
}