#include "NeuralNetwork.h"
#include <iostream>
#include "MNIST.h"
#include <iomanip>

int main(void) {
    
    try {
        std::string imagePath = "mnist/train-images.idx3-ubyte";
        std::string labelPath = "mnist/train-labels.idx1-ubyte";

        auto dataset = readImageData<float>(imagePath, labelPath);

        std::cout << "Successfully loaded dataset!\n";
        std::cout << "Total images: " << dataset.images.size() << "\n";
        std::cout << "Image dimensions: " << dataset.imageWidth << "x" << dataset.imageHeight << "\n";
        
        NeuralNetwork<float, SoftmaxCCELoss> net;

        // 28x28=784 inputs, 128 neurons, 10 outputs (digits 0-9)
        net.addLayer(std::make_unique<DenseLayer<float, ReLU_Activation>>(784, 128));
        net.addLayer(std::make_unique<DenseLayer<float, Linear_Activation>>(128, 10));

        int seed = 80085;
        std::mt19937 gen(seed);
        net.initializeAllWeights(gen);

        // training loop
        float learningRate = 0.01f;
        size_t epochs = 5;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0.0f;
            size_t totalImages = dataset.images.size();
            size_t progressInterval = 1000; // refresh progress bar every 1000 images

            for (size_t i = 0; i < totalImages; ++i) {
                // Extract input vector and target vector directly from image
                auto input = dataset.images[i].toVector<float, Vector>();
                auto target = dataset.images[i].toTargetVector<float, Vector>(10);

                // Train sample and accumulate loss
                totalLoss += net.trainSample(input, target, learningRate);
                
                // Update progress bar periodically
                if ((i + 1) % progressInterval == 0 || i + 1 == totalImages) {
                    float progress = static_cast<float>(i + 1) / totalImages;
                    int barWidth = 30;
                    int pos = static_cast<int>(barWidth * progress);

                    std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
                    for (int b = 0; b < barWidth; ++b) {
                        if (b < pos) std::cout << "=";
                        else if (b == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0f) << "% "
                            << "Loss: " << std::setprecision(4) << (totalLoss / (i + 1)) << std::flush;
                }
            }
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}