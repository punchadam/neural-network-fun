#pragma once

#include "linearAlgebra.h"
#include <random>

// idk if this is the right way to do this i'm
// just trying to get started lol

struct ReLU_Activation {
    template <typename T>
    static inline T activate(T x) {
        return x > static_cast<T>(0) ? x : static_cast<T>(0);
    }
    
    // derivative is 0 when x < 0 and 1 when x > 0 bc derivative of x is 1

    template <typename T>
    static inline T derivative_from_z(T z) {
        return z > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
    }
};

struct Sigmoid_Activation {
    template <typename T>
    static inline T activate(float x) {
        return static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(-x)));
    }
    
    // uses output z if sig = z
    template <typename T>
    static inline T derivative_from_z(T z) {
        return z * (static_cast<T>(1) - z);
    }
};

template <typename T, typename Function> // data type and activation function
class DenseLayer {
    
    static_assert(std::is_floating_point<T>::value, "Type must be floating-point");

private:
    Matrix<T> W;
    Vector<T> b;

    // store the raw input vector cause backprop needs it #someonepassedCSE205 #encapsulaysh
    Vector<T> x;
    Vector<T> z;

    // pre-allocated buffer for backward pass
    Vector<T> delta;

    Function ActivationFunction;

public:

    DenseLayer(size_t inputSize, size_t outputSize)
        : W(outputSize, inputSize), b(outputSize, T{0}), x(inputSize), z(outputSize) {} // biases init to 0 is okay
    
    void initializeWeights(std::mt19937& gen) {
        // initialize weight values using He distribution
        // std dev = sqrt(2/input size)
        const T stdDev = std::sqrt(static_cast<T>(2.0) / static_cast<T>(W.cols()));
        std::normal_distribution<T> dist(T{0}, stdDev);

        // lambda with by-reference to gen so it doesn't produce identical layers
        W.apply([&](T) { return dist(gen); });
    }
    
    // forward pass, z = W * x + b
    Vector<T> forwardPass(const Vector<T>& input) {
        x = input;
        z = (W * x) + b;  // this is fucking sick all that linear alg work paid off
        return z.apply([](T val) { return Function::activate(val); });
    }

    // backward pass, delta = dL/dz * derivative of activation(z)
    Vector<T> backwardPass(const Vector<T>& dL_da, T learningRate) {
        // self explanatory, again thank fuck i did all the work in the header
        // i needed that cause NNs are new to me so i wanted this as simple as possible:

        delta = z.apply([](T val) { return Function::derivative_from_z(val); });
        delta.hadamard(dL_da);

        Vector<T> dL_dx = delta * W;

        W -= (Matrix<T>::outer(delta, x) * learningRate);

        b -= (delta * learningRate);

        return dL_dx;
    }
};