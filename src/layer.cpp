#include "layer.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

Layer::Layer(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
    std::srand(std::time(0));

    // Initialize weights with random values
    weights.resize(output_size, std::vector<double>(input_size));
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = randomWeight();
        }
    }

    // Initialize biases with random values
    biases.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        biases[i] = randomWeight();
    }
}

// Generate a random weight in the range [-1, 1]
double Layer::randomWeight() {
    return (double)std::rand() / RAND_MAX * 2.0 - 1.0;
}

// Forward propagation: input -> weighted sum -> activation function
std::vector<double> Layer::forward(const std::vector<double>& input) {
    std::vector<double> output(output_size, 0.0);
    // Basic matrix-vector multiplication
    for (int i = 0; i < output_size; ++i) {
        double weighted_sum = 0.0;
        for (int j = 0; j < input_size; ++j) {
            weighted_sum += weights[i][j] * input[j];
        }
        weighted_sum += biases[i];
        output[i] = 1.0 / (1.0 + std::exp(-weighted_sum));
    }

    // Store the input and output for use in backpropagation
    last_input = input;
    last_output = output;

    return output;
}

// Backward propagation
std::vector<double> Layer::backward(const std::vector<double>& gradients, double learning_rate) {
    // Derivative of activation (sigmoid)
    std::vector<double> activation_derivatives(output_size);
    for (int i = 0; i < output_size; ++i) {
        double sigmoid_output = last_output[i];
        activation_derivatives[i] = sigmoid_output * (1.0 - sigmoid_output);
    }

    std::vector<std::vector<double>> weight_gradients(output_size, std::vector<double>(input_size, 0.0));
    std::vector<double> bias_gradients(output_size, 0.0);

    // Compute weight gradients and propagate error to previous layer
    std::vector<double> input_gradients(input_size, 0.0);
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            double delta = gradients[i] * activation_derivatives[i];
            weight_gradients[i][j] = delta * last_input[j];
            input_gradients[j] += weights[i][j] * delta;
        }
        bias_gradients[i] = gradients[i] * activation_derivatives[i];
    }

    // Update weights and biases using the computed gradients
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] -= learning_rate * weight_gradients[i][j];
        }
        biases[i] -= learning_rate * bias_gradients[i];
    }

    return input_gradients;  // Return the gradients to be used by the previous layer
}
