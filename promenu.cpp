#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>

using namespace std;

template <typename T>
class Matrix;

template <typename T>
class Vector {
private:
    vector<T> data;

public:
    Vector(size_t size) : data(size) {}

    void setElement(size_t index, T value) {
        data[index] = value;
    }

    T getElement(size_t index) const {
        return data[index];
    }

    size_t size() const {
        return data.size();
    }

    void display() const {
        for (size_t i = 0; i < size(); ++i) {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    Vector<T> operator+(const Vector<T>& other) const {
        if (size() != other.size()) {
            throw invalid_argument("Vector dimensions must match for addition");
        }
        Vector<T> result(size());
        for (size_t i = 0; i < size(); ++i) {
            result.setElement(i, data[i] + other.getElement(i));
        }
        return result;
    }

    Vector<T> operator-(const Vector<T>& other) const {
        if (size() != other.size()) {
            throw invalid_argument("Vector dimensions must match for subtraction");
        }
        Vector<T> result(size());
        for (size_t i = 0; i < size(); ++i) {
            result.setElement(i, data[i] - other.getElement(i));
        }
        return result;
    }

    T innerProduct(const Vector<T>& other) const {
        if (size() != other.size()) {
            throw invalid_argument("Both vectors must have the same size for inner product");
        }
        T result = 0;
        for (size_t i = 0; i < size(); ++i) {
            result += data[i] * other.getElement(i);
        }
        return result;
    }
};

template <typename T>
class Matrix {
private:
    vector<vector<T>> data;

public:
    Matrix(size_t rows, size_t cols) : data(rows, vector<T>(cols)) {}

    void setElement(size_t row, size_t col, T value) {
        data[row][col] = value;
    }

    T getElement(size_t row, size_t col) const {
        return data[row][col];
    }

    size_t numRows() const {
        return data.size();
    }

    size_t numCols() const {
        return data.empty() ? 0 : data[0].size();
    }

    void display() const {
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
        if (numRows() != other.numRows() || numCols() != other.numCols()) {
            throw invalid_argument("Matrix dimensions must match for addition");
        }
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] + other.getElement(i, j));
            }
        }
        return result;
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
        if (numRows() != other.numRows() || numCols() != other.numCols()) {
            throw invalid_argument("Matrix dimensions must match for subtraction");
        }
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] - other.getElement(i, j));
            }
        }
        return result;
    }

    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(numRows(), numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(i, j, data[i][j] * scalar);
            }
        }
        return result;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (numCols() != other.numRows()) {
            throw invalid_argument("Number of columns in first matrix must match number of rows in second matrix for multiplication");
        }
        Matrix<T> result(numRows(), other.numCols());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < other.numCols(); ++j) {
                T sum = 0;
                for (size_t k = 0; k < numCols(); ++k) {
                    sum += data[i][k] * other.getElement(k, j);
                }
                result.setElement(i, j, sum);
            }
        }
        return result;
    }

    Matrix<T> transpose() const {
        Matrix<T> result(numCols(), numRows());
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                result.setElement(j, i, data[i][j]);
            }
        }
        return result;
    }

    T norm() const {
        T sum = 0;
        for (size_t i = 0; i < numRows(); ++i) {
            for (size_t j = 0; j < numCols(); ++j) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sqrt(sum);
    }

    T determinant() const {
        if (numRows() != 2 || numCols() != 2) {
            throw invalid_argument("Determinant calculation is only supported for 2x2 matrices");
        }
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }

    vector<T> eigenValues() const {
        if (numRows() != 2 || numCols() != 2) {
            throw invalid_argument("Eigenvalues calculation is only supported for 2x2 matrices");
        }
        T a = data[0][0];
        T b = data[0][1];
        T c = data[1][0];
        T d = data[1][1];
        T discriminant = sqrt((a + d) * (a + d) - 4 * (a * d - b * c));
        T lambda1 = (a + d + discriminant) / 2;
        T lambda2 = (a + d - discriminant) / 2;
        return {lambda1, lambda2};
    }

    Matrix<T> inverse() const {
        if (numRows() != 2 || numCols() != 2) {
            throw invalid_argument("Matrix inversion is only supported for 2x2 matrices");
        }
        T det = determinant();
        if (det == 0) {
            throw invalid_argument("Matrix is singular, cannot be inverted");
        }
        Matrix<T> result(2, 2);
        result.setElement(0, 0, data[1][1] / det);
        result.setElement(0, 1, -data[0][1] / det);
        result.setElement(1, 0, -data[1][0] / det);
        result.setElement(1, 1, data[0][0] / det);
        return result;
    }
};

template <typename T>
void performVectorAddition(const Vector<T>& vec1, const Vector<T>& vec2) {
    try {
        Vector<T> result_add = vec1 + vec2;
        cout << "Vector Addition Result:" << endl;
        result_add.display();
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
    }
}

template <typename T>
void performVectorSubtraction(const Vector<T>& vec1, const Vector<T>& vec2) {
    try {
        Vector<T> result_sub = vec1 - vec2;
        cout << "Vector Subtraction Result:" << endl;
        result_sub.display();
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
    }
}

template <typename T>
void performVectorInnerProduct(const Vector<T>& vec1, const Vector<T>& vec2) {
    try {
        T innerProd = vec1.innerProduct(vec2);
        cout << "Vector Inner Product Result:" << endl;
        cout << innerProd << endl;
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
    }
}

int main() {
    int choice;
    do {
        cout << "Menu:" << endl;
        cout << "1. Matrix Operations" << endl;
        cout << "2. Vector Operations" << endl;
        cout << "3. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1: {
                // Matrix Operations
                size_t rows1, cols1, rows2, cols2;

                cout << "Enter the number of rows for matrix 1: ";
                cin >> rows1;
                cout << "Enter the number of columns for matrix 1: ";
                cin >> cols1;

                Matrix<int> mat1(rows1, cols1);
                cout << "Enter elements for matrix 1:" << endl;
                for (size_t i = 0; i < rows1; ++i) {
                    for (size_t j = 0; j < cols1; ++j) {
                        int value;
                        cout << "Enter element at position (" << i << ", " << j << "): ";
                        cin >> value;
                        mat1.setElement(i, j, value);
                    }
                }

                cout << "Matrix 1:" << endl;
                mat1.display();

                cout << "Enter the number of rows for matrix 2: ";
                cin >> rows2;
                cout << "Enter the number of columns for matrix 2: ";
                cin >> cols2;

                Matrix<int> mat2(rows2, cols2);
                cout << "Enter elements for matrix 2:" << endl;
                for (size_t i = 0; i < rows2; ++i) {
                    for (size_t j = 0; j < cols2; ++j) {
                        int value;
                        cout << "Enter element at position (" << i << ", " << j << "): ";
                        cin >> value;
                        mat2.setElement(i, j, value);
                    }
                }

                cout << "Matrix 2:" << endl;
                mat2.display();

                int matrixOperation;
                do {
                    cout << "Matrix Operation Menu:" << endl;
                    cout << "1. Matrix Addition" << endl;
                    cout << "2. Matrix Subtraction" << endl;
                    cout << "3. Scalar Multiplication" << endl;
                    cout << "4. Matrix Multiplication" << endl;
                    cout << "5. Transpose" << endl;
                    cout << "6. Norm" << endl;
                    cout << "7. Inverse" << endl;
                    cout << "8. Eigenvalues" << endl;
                    cout << "9. Determinant" << endl;
                    cout << "10. Exit to main menu" << endl;
                    cout << "Enter your choice: ";
                    cin >> matrixOperation;

                    switch (matrixOperation) {
                        case 1:
                            try {
                                Matrix<int> result_add = mat1 + mat2;
                                cout << "Matrix Addition Result:" << endl;
                                result_add.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 2:
                            try {
                                Matrix<int> result_sub = mat1 - mat2;
                                cout << "Matrix Subtraction Result:" << endl;
                                result_sub.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 3:
                            int scalar;
                            cout << "Enter the scalar value: ";
                            cin >> scalar;
                            try {
                                Matrix<int> result_mult = mat1 * scalar;
                                cout << "Scalar Multiplication Result:" << endl;
                                result_mult.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 4:
                            try {
                                Matrix<int> result_mult = mat1 * mat2;
                                cout << "Matrix Multiplication Result:" << endl;
                                result_mult.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 5:
                            try {
                                Matrix<int> result_transpose = mat1.transpose();
                                cout << "Transpose Result:" << endl;
                                result_transpose.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 6:
                            try {
                                cout << "Norm of Matrix 1: " << mat1.norm() << endl;
                                cout << "Norm of Matrix 2: " << mat2.norm() << endl;
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 7:
                            try {
                                Matrix<int> result_inverse = mat1.inverse();
                                cout << "Inverse of Matrix 1:" << endl;
                                result_inverse.display();
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 8:
                            try {
                                vector<int> eigenvals1 = mat1.eigenValues();
                                vector<int> eigenvals2 = mat2.eigenValues();
                                cout << "Eigenvalues of Matrix 1: ";
                                for (int val : eigenvals1) {
                                    cout << val << " ";
                                }
                                cout << endl;
                                cout << "Eigenvalues of Matrix 2: ";
                                for (int val : eigenvals2) {
                                    cout << val << " ";
                                }
                                cout << endl;
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 9:
                            try {
                                cout << "Determinant of Matrix 1: " << mat1.determinant() << endl;
                                cout << "Determinant of Matrix 2: " << mat2.determinant() << endl;
                            } catch (const invalid_argument& e) {
                                cerr << e.what() << endl;
                            }
                            break;
                        case 10:
                            break;
                        default:
                            cout << "Invalid choice. Please enter a number between 1 and 10." << endl;
                    }
                } while (matrixOperation != 10);
                break;
            }
            case 2: {
                // Vector Operations
                size_t size1, size2;

                cout << "Enter the size of vector 1: ";
                cin >> size1;

                Vector<int> vec1(size1);
                cout << "Enter elements for vector 1:" << endl;
                for (size_t i = 0; i < size1; ++i) {
                    int value;
                    cout << "Enter element at position " << i << ": ";
                    cin >> value;
                    vec1.setElement(i, value);
                }

                cout << "Vector 1:" << endl;
                vec1.display();

                cout << "Enter the size of vector 2: ";
                cin >> size2;

                Vector<int> vec2(size2);
                cout << "Enter elements for vector 2:" << endl;
                for (size_t i = 0; i < size2; ++i) {
                    int value;
                    cout << "Enter element at position " << i << ": ";
                    cin >> value;
                    vec2.setElement(i, value);
                }

                cout << "Vector 2:" << endl;
                vec2.display();

                int vectorOperation;
                do {
                    cout << "Vector Operation Menu:" << endl;
                    cout << "1. Vector Addition" << endl;
                    cout << "2. Vector Subtraction" << endl;
                    cout << "3. Vector Inner Product" << endl;
                    cout << "4. Exit to main menu" << endl;
                    cout << "Enter your choice: ";
                    cin >> vectorOperation;

                    switch (vectorOperation) {
                        case 1:
                            performVectorAddition(vec1, vec2);
                            break;
                        case 2:
                            performVectorSubtraction(vec1, vec2);
                            break;
                        case 3:
                            performVectorInnerProduct(vec1, vec2);
                            break;
                        case 4:
                            break;
                        default:
                            cout << "Invalid choice. Please enter a number between 1 and 4." << endl;
                    }
                } while (vectorOperation != 4);
                break;
            }
            case 3:
                cout << "Exiting program." << endl;
                break;
            default:
                cout << "Invalid choice. Please enter a number between 1 and 3." << endl;
        }
    } while (choice != 3);

    return 0;
}