#include "NeuralNetwork.h"
#include <iostream>
#include "MNIST.h"

int main(void) {
    
    readFile("test.txt");

    /*
    NeuralNetwork<float, SoftmaxCCELoss> net;

    // 28x28=784 inputs, 128 neurons, 10 outputs (digits 0-9)
    net.addLayer(std::make_unique<DenseLayer<float, ReLU_Activation>>(784, 128));
    net.addLayer(std::make_unique<DenseLayer<float, Sigmoid_Activation>>(128, 10));

    int seed = 80085;
    std::mt19937 gen(seed);
    net.initializeAllWeights(gen);

    // training loop
    size_t epochs = 10;
    float learningRate = 0.01f;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float totalLoss = 0.0f;
        for (size_t i = 0; i < trainingImages.size(); i++) {
            totalLoss += net.trainSample(trainingImages[i], targetVectors[i], learningRate);
        }
        std::cout << "Epoch " << epoch << " Loss: " << totalLoss / trainingImages.size() << "\n";
    }
    */

    return 0;
}