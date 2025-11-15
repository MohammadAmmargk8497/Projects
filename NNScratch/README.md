# Neural Network from Scratch with Real-Time Visualization

This project provides a hands-on implementation of a simple two-layer neural network from scratch using NumPy. It also includes a web-based visualization that shows the network learning in real-time. The neural network is trained on the MNIST dataset of handwritten digits.

## How it Works

The project consists of two main parts:
1.  **The Neural Network**: Implemented in Python using NumPy. It learns to classify handwritten digits.
2.  **The Visualization**: A Flask web application that visualizes the network's structure, activations, and learning process in real-time.

The neural network is a fully connected two-layer network that uses the Sigmoid activation function. It is trained using batch gradient descent.

## The Neural Network Architecture

The network has the following architecture:
- **Input Layer**: 784 neurons, corresponding to the 28x28 pixels of the MNIST images.
- **Hidden Layer**: 100 neurons with Sigmoid activation.
- **Output Layer**: 10 neurons with Sigmoid activation, corresponding to the 10 digit classes (0-9).

The weights and biases are initialized with random values.

```python
def initialize_parameters():
    W1 = np.random.randn(100, 784) * 0.01
    b1 = np.zeros((100, 1))
    W2 = np.random.randn(10, 100) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2
```

## Forward Propagation

During the forward pass, the input data is fed through the network to produce an output.

1.  **From Input to Hidden Layer**: The input `X` is multiplied by the weights `W1` and the bias `b1` is added. The result is passed through the Sigmoid activation function to get the activation `A1`.
2.  **From Hidden to Output Layer**: The activation `A1` from the hidden layer is multiplied by the weights `W2` and the bias `b2` is added. The result is passed through the Sigmoid activation function again to get the final output `A2`.

Here is the implementation of the forward pass from `app.py`:

```python
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
```

## Backward Propagation (Backpropagation)

Backpropagation is used to calculate the gradients of the loss function with respect to the weights and biases. These gradients are then used to update the parameters. The loss is calculated using the Mean Squared Error between the predicted output and the one-hot encoded true labels.

1.  **Calculate Output Layer Error**: The error in the output layer (`dZ2`) is calculated as the difference between the predicted output `A2` and the one-hot encoded true labels `one_hot_Y`.
2.  **Calculate Gradients for W2 and b2**: The gradients for the weights (`dW2`) and biases (`db2`) of the output layer are calculated based on `dZ2` and the activations of the hidden layer `A1`.
3.  **Calculate Hidden Layer Error**: The error is propagated back to the hidden layer (`dZ1`).
4.  **Calculate Gradients for W1 and b1**: The gradients for the weights (`dW1`) and biases (`db1`) of the hidden layer are calculated based on `dZ1` and the input `X`.

Here is the implementation of the backward pass from `app.py`:

```python
def backward_propagation(W2, Z1, A1, A2, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)
    
    dZ2 = 2 * (A2 - one_hot_Y)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
```

## Parameter Updates

The weights and biases are updated using gradient descent. The calculated gradients are multiplied by a learning rate (`alpha`) and subtracted from the current parameters.

```python
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2
```

## Real-Time Visualization

The project uses Flask and Flask-SocketIO to create a web-based visualization of the neural network. When the training starts, the backend (`app.py`) continuously sends the network's state (weights, biases, activations, gradients) to the frontend. The frontend, built with HTML, CSS, and JavaScript, visualizes this data in real-time.

## How to Run the Project

To run the project, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    This project requires the following Python libraries:
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `flask`
    - `flask-socketio`

    You can install them using pip:
    ```bash
    pip install numpy pandas scikit-learn flask flask-socketio
    ```

3.  **Download the dataset**:
    The project uses the MNIST dataset. You can download it from Kaggle:
    [https://www.kaggle.com/competitions/digit-recognizer/data](https://www.kaggle.com/competitions/digit-recognizer/data)

    Download the `train.csv` file and save it.

4.  **Update the dataset path**:
    The path to the dataset is hardcoded in `app.py`. You need to update the `CSV_PATH` variable to point to the location of your `train.csv` file.

    In `app.py`, find this line:
    ```python
    CSV_PATH = '/Users/ammar/Downloads/mnist_train.csv'
    ```
    And change it to the correct path on your system. For example:
    ```python
    CSV_PATH = '/path/to/your/train.csv'
    ```

5.  **Run the application**:
    ```bash
    python app.py
    ```

6.  **View the visualization**:
    Open your web browser and go to `http://localhost:5000`. You should see the neural network visualization. Click the "Start Training" button to see the network learn in real-time.
