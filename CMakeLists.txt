cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

include_directories(include)

add_executable(NeuralNet main.cpp src/layer.cpp src/neural_network.cpp src/activation.cpp src/loss.cpp src/utils.cpp)
