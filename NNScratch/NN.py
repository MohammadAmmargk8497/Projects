import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('/Users/zunairaasif/Downloads/train.csv')

data.head()

data = np.array(data)
m, n = data.shape
np.random .shuffle(data)

data_dev = data[0:1000].T
X_dev = data_dev[1:n]
Y_dev = data_dev[0]

data_train = data[1000:m].T
X_train = data_train[1:n]
Y_train = data_train[0]

Ns = MinMaxScaler()
X_train = Ns.fit_transform(X_train)


def init_params():
    W1 = np.random.randn(100,784) 
    b1 = np.random.randn(100,1) 
    
    W2 = np.random.randn(10, 100) 
    b2 = np.random.randn(10,1)
    return W1, b1, W2, b2


def Sigmoid(Z):
    return  1.0/(1.0+np.exp(-Z))

def deriv_sigmoid(Z):
    return Sigmoid(Z)*(1-Sigmoid(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Sigmoid(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = Sigmoid(Z2)
    
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(W1,  W2, Z1, Z2, A1, A2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    dZ2 = 2*(A2 - one_hot_Y)
    dW2 = 1/m*dZ2.dot(A1.T)
    db2 = 1/m*np.sum(dZ2,1)
    
    dZ1 = W2.T.dot(dZ2)*deriv_sigmoid(Z1)
    dW1 = 1/m*dZ1.dot(X.T)
    db1 = 1/m*np.sum(dZ1,1)
    
    return  dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    
    W2 = W2 - alpha*dW2
    
    b1 = b1 - alpha*np.reshape(db1, (100,1))
    
    b2 = b2 - alpha*np.reshape(db2, (10,1))
    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations):  
        
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    
        dW1, db1, dW2, db2 = back_prop(W1,  W2, Z1, Z2, A1, A2, X, Y)
        
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 3)