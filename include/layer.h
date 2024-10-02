#pragma oncde

#include <vector>

class Layer{
public:
    Layer(int input_size, int output_size);
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& gradients, double learning_rate);
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    int input_size;
    int output_size;

    std::vector<double> last_input;
    std::vector<double> last_output;

    double randomWeight();
};
