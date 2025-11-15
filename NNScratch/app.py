import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import threading
import os

# ==========================================
# DEBUGGING & PATH RESOLUTION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
socketio = SocketIO(app, cors_allowed_origins="*")

# Verify directories exist
if not os.path.exists(app.template_folder):
    print(f"❌ ERROR: Template folder not found at {app.template_folder}")
    exit(1)

if not os.path.exists(app.static_folder):
    print(f"❌ ERROR: Static folder not found at {app.static_folder}")
    exit(1)

print(f"✅ Templates: {app.template_folder}")
print(f"✅ Static: {app.static_folder}")

# ==========================================
# NEURAL NETWORK CODE (Unchanged)
# ==========================================

def load_and_preprocess_data(csv_path):
    """Load and preprocess MNIST data"""
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    data = np.array(data)
    np.random.shuffle(data)
    
    # Use smaller dataset for faster animation
    data = data[:5000]  
    
    data_val = data[0:500].T
    X_val = data_val[1:]
    Y_val = data_val[0]
    
    data_train = data[500:].T
    X_train = data_train[1:]
    Y_train = data_train[0]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_val = scaler.transform(X_val.T).T
    
    print(f"✅ Data loaded: {X_train.shape[1]} training samples, {X_val.shape[1]} validation samples")
    return X_train, Y_train, X_val, Y_val

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def initialize_parameters():
    W1 = np.random.randn(100, 784) * 0.01
    b1 = np.zeros((100, 1))
    W2 = np.random.randn(10, 100) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def compute_loss(A2, Y):
    m = Y.shape[0]
    one_hot_Y = one_hot_encode(Y)
    return np.mean(np.square(A2 - one_hot_Y))

def compute_accuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    return np.sum(predictions == Y) / Y.size

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

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

# ==========================================
# ANIMATION-AWARE TRAINING LOOP (FIXED!)
# ==========================================

def animated_training_loop(X_train, Y_train, X_val, Y_val, iterations, learning_rate, socketio):
    """Modified training loop that yields animation frames"""
    
    W1, b1, W2, b2 = initialize_parameters()
    
    # Send initial network state
    
    emit_network_state(socketio, W1, b1, W2, b2, None, None, None, None, 0, 0, 0, phase="init")
    time.sleep(1)
    
    for i in range(iterations):
        print(f"🔄 Iteration {i+1}/{iterations}")
        
        # Forward Propagation Phase
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_train)
        loss = compute_loss(A2, Y_train)
        accuracy = compute_accuracy(A2, Y_train)
        
        # Send forward propagation animation
        true_class = Y_train[0] if Y_train.size > 0 else None
        emit_network_state(socketio, W1, b1, W2, b2, Z1, A1, Z2, A2, 
                          loss, accuracy, i, phase="forward", true_class=true_class)
        time.sleep(0.3)  # Shorter delay for faster animation
        
        # Backward Propagation Phase
        dW1, db1, dW2, db2 = backward_propagation(W2, Z1, A1, A2, X_train, Y_train)
        
        # Send backward propagation animation
        emit_network_state(socketio, W1, b1, W2, b2, Z1, A1, Z2, A2,
                          loss, accuracy, i, phase="backward", 
                          dW1=dW1, db1=db1, dW2=dW2, db2=db2)
        time.sleep(0.3)
        
        # Update Phase
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        # Send update phase
        emit_network_state(socketio, W1, b1, W2, b2, Z1, A1, Z2, A2,
                          loss, accuracy, i, phase="update")
        
        # Validation metrics
        if i % 10 == 0:
            _, _, _, A2_val = forward_propagation(W1, b1, W2, b2, X_val)
            val_loss = compute_loss(A2_val, Y_val)
            val_accuracy = compute_accuracy(A2_val, Y_val)
            
            # ✅ CORRECTED: Use socketio.emit() not emit()
            socketio.emit('validation_metrics', {
                'iteration': i,
                'train_loss': float(loss),
                'train_accuracy': float(accuracy),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy)
            })

def emit_network_state(socketio, W1, b1, W2, b2, Z1, A1, Z2, A2, loss, accuracy, iteration, 
                      phase="forward", dW1=None, db1=None, dW2=None, db2=None, true_class=None):
    """Package and send network state to frontend"""
    
    # Downsample weights for visualization
    W1_sample = W1[::10, ::50].tolist()
    W2_sample = W2[::2, ::10].tolist()
    
    state = {
        'iteration': iteration,
        'phase': phase,
        'loss': float(loss) if loss is not None else 0,
        'accuracy': float(accuracy) if accuracy is not None else 0,
        'weights': {
            'W1': W1_sample,
            'W2': W2_sample
        },
        'biases': {
            'b1': b1[::10].tolist(),
            'b2': b2.tolist()
        },
        'trueClass': int(true_class) if true_class is not None else None  # Add this
    }
    
    if A1 is not None:
        state['activations'] = {
            'A1': A1[::10, 0].tolist(),
            'A2': A2[:, 0].tolist()
        }
    
    if phase == "backward" and dW1 is not None:
        state['gradients'] = {
            'dW1': dW1[::10, ::50].tolist(),
            'dW2': dW2[::2, ::10].tolist(),
            'db1': db1[::10].tolist(),
            'db2': db2.tolist()
        }
    
    # ✅ Use socketio.emit() directly
    socketio.emit('network_update', state)

# ==========================================
# FLASK ROUTES
# ==========================================

CSV_PATH = '/Users/ammar/Downloads/mnist_train.csv'

# Load data at startup (with error handling)
try:
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(CSV_PATH)
    print(f"✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    print("Creating dummy data for testing...")
    X_train, Y_train = np.random.randn(784, 100), np.random.randint(0, 10, 100)
    X_val, Y_val = np.random.randn(784, 50), np.random.randint(0, 10, 50)

@app.route('/')
def index():
    """Serve the visualization page"""
    return render_template('index.html')

@socketio.on('start_training')
def handle_start_training():
    """Start training in a background thread"""
    print("🚀 Starting training...")
    
    def run_training():
        animated_training_loop(
            X_train, Y_train,
            X_val, Y_val,
            iterations=500,  # Reduced for demo
            learning_rate=0.5,
            socketio=socketio
        )
        print("✅ Training complete!")
    
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎮 Neural Network Visualizer")
    print("="*50)
    print("🌐 Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    
    socketio.run(app, debug=True, port=5000)