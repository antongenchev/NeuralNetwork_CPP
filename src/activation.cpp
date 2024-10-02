#include "activation.h"
#include <cmath>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x){
    double sigmoid_value = sigmoid(x);
    return sigmoid_value * (1.0 - sigmoid_value);
};