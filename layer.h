#pragma once

#include "linearAlgebra.h"
#include <random>
#include <memory>

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

// for continuous output
struct MeanSquaredLoss {
    // returns a scalar T of the avg squared difference across all elements
    template <typename T>
    static inline T calculateLoss(const Vector<T>& predicted, const Vector<T>& target) {
        Vector<T> diff = predicted - target;
        diff.apply([](T val) { return val * val; });    // squares each value in place
        T sum = static_cast<T>(0);
        diff.apply([&sum](T val) { sum += val; return val; });  // sums all of the squares into sum
        return sum / static_cast<T>(diff.size());   // 1/N to get Mean Squared Error
    }

    // returns the initial error vector<T> given a raw target vector
    template <typename T>
    static inline Vector<T> calculateGradient(const Vector<T>& predicted, const Vector<T>& target) {
        Vector<T> diff = predicted - target;
        diff *= static_cast<T>(2) / static_cast<T>(diff.size());    // dL/dy_predicted = 2/N(y_predicted - y_target)
        return diff;
    }
};

// for classification output
struct SoftmaxCCELoss {

    template <typename T>
    static inline Vector<T> applySoftmaxFunction(const Vector<T>& input) {
        Vector<T> softmaxOutput = input;
        T max = std::numeric_limits<T>::lowest();
        softmaxOutput.apply([&max](T val) { max = std::max(val, max); return val; });   // find maximum logit value
        softmaxOutput.apply([&max](T val) { return val - max; });   // subtract max from every logit
        
        T sumExp = static_cast<T>(0);
        T e_val;   // init here to avoid a bunch of allocations
        softmaxOutput.apply([&sumExp, &e_val](T val) { e_val = std::exp(val); sumExp += e_val; return e_val; });
        softmaxOutput.apply([&sumExp](T val) { return val / sumExp; });   // softmaxOutput is now a finished probability vector
        return softmaxOutput;
    }

    // applies softmax, then calculates the categorical cross-entropy
    template <typename T>
    static inline T calculateLoss(const Vector<T>& predicted, const Vector<T>& target) {
        Vector<T> softmaxOutput = applySoftmaxFunction(predicted);
        
        T tinyNumberToAvoidNaNforSmallFloats = static_cast<T>(1e-7);
        T CCELoss = 0;
        for (size_t i = 0; i < softmaxOutput.size(); i++) {
            CCELoss += target[i] * std::log(softmaxOutput[i] + tinyNumberToAvoidNaNforSmallFloats);
        }
        CCELoss *= static_cast<T>(-1);

        return CCELoss;
    }
    
    template <typename T>
    static inline Vector<T> calculateGradient(const Vector<T>& predicted, const Vector<T>& target) {
        Vector<T> softmaxOutput = applySoftmaxFunction(predicted);
        
        // derivative simplifies to just probabiliies - target
        softmaxOutput -= target;
        return softmaxOutput;
    }
};

// common layer interface my professor would be happy
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    virtual void initializeWeights(std::mt19937& gen) = 0;

    virtual Vector<T> forwardPass(const Vector<T>& input) = 0;

    virtual Vector<T> backwardPass(const Vector<T>& dL_da, T learningRate) = 0;
};

template <typename T, typename Function> // data type and activation function
class DenseLayer : public Layer<T> {
    
    static_assert(std::is_floating_point<T>::value, "Type must be floating-point");

private:
    Matrix<T> W;
    Vector<T> b;

    // store the raw input vector and delta buffer cause backprop needs it #someonepassedCSE205 #encapsulaysh
    Vector<T> x;
    Vector<T> z;
    Vector<T> delta;

    Function ActivationFunction;

public:

    DenseLayer(size_t inputSize, size_t outputSize)
        : W(outputSize, inputSize), b(outputSize, T{0}), x(inputSize), z(outputSize) {} // biases init to 0 is okay
    
    // layer doesn't own the mt object so that layers don't come out the same or similar
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

    // backward pass, delta = dL/dz * derivative of activation (z)
    Vector<T> backwardPass(const Vector<T>& dL_dz, T learningRate) {
        // self explanatory, again thank fuck i did all the work in the header
        // i needed that cause NNs are new to me so i wanted this as simple as possible:

        delta = z.apply([](T val) { return Function::derivative_from_z(val); });
        delta.hadamard(dL_dz);

        Vector<T> dL_dx = delta * W;

        W -= (Matrix<T>::outer(delta, x) * learningRate);

        b -= (delta * learningRate);

        return dL_dx;
    }
};