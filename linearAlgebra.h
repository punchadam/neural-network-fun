#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits> 
#include <initializer_list>
#include <type_traits>

template <typename T>
class Vector {
private:
    std::vector<T> m_data;

public:
    Vector() = default;
    explicit Vector(size_t size, T initialValue = T{}) : m_data(size, initialValue) {}
    Vector(std::initializer_list<T> list) : m_data(list) {}

    size_t size() const {
        return m_data.size();
    }

    // mutable access
    T& operator[](size_t index) {
        return this->m_data[index];
    }

    // const access
    const T& operator[](size_t index) const {
        return this->m_data[index];
    }

    // magnitude function
    // weird type stuff to maintain accuracy
    auto magnitude() const {
        using ReturnType = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        ReturnType sum = ReturnType{0};
        for (const T& val : m_data) {
            sum += static_cast<ReturnType>(val) * static_cast<ReturnType>(val);
        }
        return std::sqrt(sum);
    }

    // vector + vector
    Vector<T> operator+(const Vector<T>& other) const {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for addition");
        }
        Vector<T> result(this->size());
        for (size_t i = 0; i < this->size(); i++) {
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    // in-place vector += vector
    Vector<T>& operator+=(const Vector<T>& other) {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for addition");
        }
        for (size_t i = 0; i < this->size(); i++) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    // vector - vector
    Vector<T> operator-(const Vector<T>& other) const {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for subtraction");
        }
        Vector<T> result(this->size());
        for (size_t i = 0; i < this->size(); i++) {
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }

    // in-place vector -= vector
    Vector<T>& operator-=(const Vector<T>& other) {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for subtraction");
        }
        for (size_t i = 0; i < this->size(); i++) {
            (*this)[i] -= other[i];
        }
        return *this;
    }

    // vector * vector (dot product)
    T operator*(const Vector<T>& other) const {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for dot product");
        }
        T result = T{};
        for (size_t i = 0; i < this->size(); i++) {
            result += (*this)[i] * other[i];
        }
        return result;
    }

    // vector * scalar
    Vector<T> operator*(T scalar) const {
        Vector<T> result(this->size());
        for (size_t i = 0; i < this->size(); i++) {
            result[i] = (*this)[i] * scalar;
        }
        return result;
    }

    // in-place vector *= scalar
    Vector<T>& operator*=(T scalar) {
        for (size_t i = 0; i < this->size(); i++) {
            (*this)[i] *= scalar;
        }
        return *this;
    }

    // scalar * vector
    friend Vector<T> operator*(T scalar, const Vector<T>& vector) {
        return vector * scalar;
    }

    // vector hadamard product
    static Vector<T> hadamard(const Vector<T>& v1, const Vector<T>& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vector sizes must match for hadamard product");
        }
        Vector<T> result(v1.size());
        for (size_t i = 0; i < v1.size(); i++) {
            result[i] = v1[i] * v2[i];
        }
        return result;
    }

    // in-place vector hadamard product
    Vector<T>& hadamard(const Vector<T>& other) {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector sizes must match for hadamard product");
        }
        for (size_t i = 0; i < this->size(); i++) {
            (*this)[i] *= other[i];
        }
        return *this;
    }

    // element-wise function
    template <typename Function>
    Vector<T> apply(Function f) const {
        Vector<T> result(this->size());
        for (size_t i = 0; i < this->size(); i++) {
            result[i] = f(this->m_data[i]);
        }
        return result;
    }

    // in-place element-wise function
    template <typename Function>
    Vector<T>& apply(Function f) {
        for (size_t i = 0; i < this->size(); i++) {
            this->m_data[i] = f(this->m_data[i]);
        }
        return *this;
    }
};

template <typename T>
class Matrix {
private:
    size_t m_rows;
    size_t m_cols;
    bool m_isTransposed;
    std::vector<T> m_data;  // flat storage

public:
    Matrix(size_t rows, size_t cols, T initialValue = T{})
        : m_rows(rows), m_cols(cols), m_isTransposed(false), m_data(rows * cols, initialValue) {}

    size_t rows() const {
        return m_isTransposed ? m_cols : m_rows;
    }
    size_t cols() const {
        return m_isTransposed ? m_rows : m_cols;
    }

    // transposes the original matrix in-place
    Matrix<T>& transpose() {
        m_isTransposed = !m_isTransposed;
        return *this;
    }

    // returns a new matrix that is the transpose of the original
    Matrix<T> transposed() const {
        Matrix<T> copy = *this;
        copy.transpose();
        return copy;
    }

    // mutable access
    T& operator[](size_t row, size_t col) {
        if (row >= this->rows() || col >= this->cols()) {
            throw std::out_of_range("Matrix access index out of bounds");
        }
        size_t actualRow = m_isTransposed ? col : row;
        size_t actualCol = m_isTransposed ? row : col;
        return m_data[actualRow * m_cols + actualCol];
    }

    // const access
    const T& operator[](size_t row, size_t col) const {
        if (row >= this->rows() || col >= this->cols()) {
            throw std::out_of_range("Matrix access index out of bounds");
        }
        size_t actualRow = m_isTransposed ? col : row;
        size_t actualCol = m_isTransposed ? row : col;
        return m_data[actualRow * m_cols + actualCol];
    }

    // matrix + matrix
    Matrix<T> operator+(const Matrix<T>& other) const {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        Matrix<T> result(this->rows(), this->cols(), T{0});
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i, j] = (*this)[i, j] + other[i, j];
            }
        }
        return result;
    }

    // in-place matrix += matrix
    Matrix<T>& operator+=(const Matrix<T>& other) {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] += other[i, j];
            }
        }
        return *this;
    }

    // matrix - matrix
    Matrix<T> operator-(const Matrix<T>& other) const {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        Matrix<T> result(this->rows(), this->cols(), T{0});
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i, j] = (*this)[i, j] - other[i, j];
            }
        }
        return result;
    }

    // in-place matrix -= matrix
    Matrix<T>& operator-=(const Matrix<T>& other) {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] -= other[i, j];
            }
        }
        return *this;
    }

    // matrix + vector (broadcasting adds the vector across every row)
    Matrix<T> operator+(const Vector<T>& vector) const {
        if (this->cols() != vector.size()) {
            throw std::invalid_argument("Matrix columns must equal vector size for broadcasting");
        }
        Matrix<T> result(this->rows(), this->cols());
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i, j] = (*this)[i, j] + vector[j];
            }
        }
        return result;
    }

    // vector + matrix
    friend Matrix<T> operator+(const Vector<T>& vector, const Matrix<T>& matrix) {
        if (matrix.cols() != vector.size()) {
            throw std::invalid_argument("Matrix columns must equal vector size for broadcasting");
        }
        return matrix + vector;
    }

    // in-place matrix += vector
    Matrix<T>& operator+=(const Vector<T>& vector) {
        if (this->cols() != vector.size()) {
            throw std::invalid_argument("Matrix columns must equal vector size for broadcasting");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] += vector[j];
            }
        }
        return *this;
    }

    // matrix * matrix
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (this->cols() != other.rows()) {
            throw std::invalid_argument("Inner dimensions of matrices must match for multiplication");
        }
        Matrix<T> result(this->rows(), other.cols(), T{0});

        for (size_t i = 0; i < result.rows(); i++) {
            for (size_t j = 0; j < result.cols(); j++) {
                for (size_t k = 0; k < this->cols(); k++) {
                    result[i, j] += (*this)[i, k] * other[k, j];
                }
            }
        }
        return result;
    }

    // in-place matrix *= matrix
    Matrix<T>& operator*=(const Matrix<T>& other) {
        *this = *this * other;
        // this just uses the above method and is the one exception to the memory-efficient in-place ops
        // it should be noted that there is no way to just compute this literally in-place without
        // extra mem allocation because parts of the equation are later dependent on values that get
        // modified during the multiplication, so a temp matrix is inevitable
        return *this;
    }

    // matrix * vector
    Vector<T> operator*(const Vector<T>& vector) const {
        if (this->cols() != vector.size()) {
            throw std::invalid_argument("Matrix columns must match vector size for multiplication");
        }
        Vector<T> result(this->rows(), T{0});
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i] += (*this)[i, j] * vector[j];
            }
        }
        return result;
    }

    // row vector * matrix
    friend Vector<T> operator*(const Vector<T>& vector, const Matrix<T>& matrix) {
        if (vector.size() != matrix.rows()) {
            throw std::invalid_argument("Vector size must match matrix rows");
        }

        Vector<T> result(matrix.cols(), T{0});
        for (size_t j = 0; j < matrix.cols(); j++) {
            for (size_t i = 0; i < matrix.rows(); i++) {
                result[j] += vector[i] * matrix[i, j];
            }
        }
        return result;
    }

    // matrix * scalar
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(this->rows(), this->cols());
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i, j] = scalar * (*this)[i, j];
            }
        }
        return result;
    }

    // in-place matrix *= scalar
    Matrix<T>& operator*=(T scalar) {
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] *= scalar;
            }
        }
        return *this;
    }

    // scalar * matrix
    friend Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
        return matrix * scalar;
    }

    // matrix hadamard product
    static Matrix<T> hadamard(const Matrix<T>& m1, const Matrix<T>& m2) {
        if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
            throw std::invalid_argument("Matrix sizes must match for hadamard product");
        }
        Matrix<T> result(m1.rows(), m1.cols());
        for (size_t i = 0; i < m1.rows(); i++) {
            for (size_t j = 0; j < m1.cols(); j++) {
                result[i, j] = m1[i, j] * m2[i, j];
            }
        }
        return result;
    }

    // in-place matrix hadamard product
    Matrix<T>& hadamard(const Matrix<T>& other) {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument("Matrix sizes must match for hadamard product");
        }
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] *= other[i, j];
            }
        }
        return *this;
    }

    // outer product
    static Matrix<T> outer(const Vector<T>& v1, const Vector<T>& v2) {
        Matrix<T> result(v1.size(), v2.size());
        for (size_t i = 0; i < result.rows(); i++) {
            for (size_t j = 0; j < result.cols(); j++) {
                result[i, j] = v1[i] * v2[j];
            }
        }
        return result;
    }

    // element-wise function
    template <typename Function>
    Matrix<T> apply(Function f) const {
        Matrix<T> result(this->rows(), this->cols());
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                result[i, j] = f((*this)[i, j]);
            }
        }
        return result;
    }

    // in-place element-wise function
    template <typename Function>
    Matrix<T>& apply(Function f) {
        for (size_t i = 0; i < this->rows(); i++) {
            for (size_t j = 0; j < this->cols(); j++) {
                (*this)[i, j] = f((*this)[i, j]);
            }
        }
        return *this;
    }
    
};