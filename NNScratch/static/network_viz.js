console.log("🎨 Network visualization script loaded");

let socket;
let networkState = null;
let isPaused = false;
let stepMode = false;
let pendingStep = false;

// ==========================================
// P5.JS VISUALIZATION
// ==========================================

let sketch = function(p) {
    let animationProgress = 0;
    let currentPhase = 'ready';
    let targetState = null;
    
    p.setup = function() {
        let canvas = p.createCanvas(900, 600);
        canvas.parent('network-canvas');
        p.textAlign(p.CENTER, p.CENTER);
        p.textSize(12);
        p.frameRate(30);
        console.log("✅ P5.js canvas initialized");
    };
    
    p.draw = function() {
        p.background(15, 15, 35); // Dark background
        
        if (!networkState) {
            p.fill(255);
            p.textSize(24);
            p.text("Waiting for training data...", p.width/2, p.height/2);
            return;
        }
        
        // Smooth animation transitions
        if (targetState && animationProgress < 1) {
            animationProgress += 0.08;
        } else {
            animationProgress = 0;
        }
        
        // Draw network layers
        drawInputLayer(p);
        drawHiddenLayer(p);
        drawOutputLayer(p);
        
        // Draw connections with proper weight scaling
        drawConnections(p);
        
        // Draw animation phase indicators
        drawPhaseAnimation(p);
        
        // Draw legend and info
        drawInfoBox(p);
    };
    
    function drawInputLayer(p) {
        // Input layer (showing 16 representative neurons)
        let x = 100;
        p.fill(100, 150, 255, 180);
        p.stroke(255);
        p.strokeWeight(1);
        
        for (let i = 0; i < 16; i++) {
            let y = p.map(i, 0, 15, 80, p.height - 80);
            p.circle(x, y, 20);
            p.fill(255);
            p.textSize(9);
            p.text(i, x, y);
            p.fill(100, 150, 255, 180);
        }
        
        p.fill(255);
        p.textSize(14);
        p.text("Input Layer", x, 30);
        p.text("784 neurons", x, 50);
    }
    
    function drawHiddenLayer(p) {
        if (!networkState.activations || !networkState.activations.A1) {
            // Draw placeholder neurons
            p.fill(100, 100, 100, 100);
            p.stroke(150);
            let x = 400;
            for (let i = 0; i < 10; i++) {
                let y = p.map(i, 0, 9, 100, p.height - 100);
                p.circle(x, y, 25);
            }
        } else {
            // Draw activated neurons
            let x = 400;
            p.stroke(255, 255, 0);
            p.strokeWeight(2);
            
            for (let i = 0; i < networkState.activations.A1.length; i++) {
                let activation = networkState.activations.A1[i];
                // Map activation to color: dark blue (0) → bright cyan (1)
                let brightness = p.map(activation, 0, 1, 50, 255);
                p.fill(0, brightness, brightness, 200);
                
                let y = p.map(i, 0, networkState.activations.A1.length, 100, p.height - 100);
                p.circle(x, y, 25);
                
                // Show activation value
                p.fill(255);
                p.textSize(8);
                p.text(activation.toFixed(2), x, y + 35);
                p.noFill();
            }
        }
        
        p.fill(255);
        p.textSize(14);
        p.text("Hidden Layer", 400, 30);
        p.text("100 neurons", 400, 50);
    }
    
    function drawOutputLayer(p) {
    if (!networkState.activations || !networkState.activations.A2) {
        // Draw placeholder neurons
        p.fill(100, 100, 100, 100);
        p.stroke(150);
        let x = 700;
        for (let i = 0; i < 10; i++) {
            let y = p.map(i, 0, 9, 150, p.height - 150);
            p.circle(x, y, 30);
        }
        return;
    }

    // Find predicted class
    let x = 700;
    let activations = networkState.activations.A2;
    let maxActivation = Math.max(...activations);
    let predictedClass = activations.indexOf(maxActivation);
    
    // Find true class (if available in networkState)
    let trueClass = networkState.trueClass !== undefined ? networkState.trueClass : null;
    
    // Draw output neurons
    for (let i = 0; i < activations.length; i++) {
        let activation = activations[i];
        let y = p.map(i, 0, activations.length, 150, p.height - 150);
        
        // Determine colors
        let fillColor, strokeColor, strokeThickness;
        
        if (i === predictedClass) {
            // PREDICTED CLASS - Highlight in bright green
            fillColor = p.color(0, 255, 0, 220); // Bright green
            strokeColor = p.color(0, 200, 0);
            strokeThickness = 4;
        } else {
            // Other classes - Normal colors
            let brightness = p.map(activation, 0, 1, 80, 255);
            fillColor = p.color(brightness, brightness * 0.7, 0, 180);
            strokeColor = p.color(255, 150, 0);
            strokeThickness = 2;
        }
        
        // Draw neuron with glow effect for predicted class
        if (i === predictedClass) {
            // Draw glow
            p.drawingContext.shadowBlur = 20;
            p.drawingContext.shadowColor = 'rgba(0, 255, 0, 0.8)';
        } else {
            p.drawingContext.shadowBlur = 0;
        }
        
        p.fill(fillColor);
        p.stroke(strokeColor);
        p.strokeWeight(strokeThickness);
        p.circle(x, y, 35); // Slightly larger for predicted
        
        // Add a star or marker for the true class
        if (i === trueClass && trueClass !== null) {
            p.drawingContext.shadowBlur = 15;
            p.drawingContext.shadowColor = 'rgba(255, 255, 0, 0.9)';
            p.fill(255, 255, 0);
            p.textSize(20);
            p.text('★', x, y - 45); // Star above the true class
            p.drawingContext.shadowBlur = 0;
        }
        
        // Show activation value
        p.fill(255);
        p.textSize(10);
        p.text(activation.toFixed(3), x, y + 45);
        
        // Show class label
        p.textSize(16);
        p.text(i, x, y);
    }
    
    // Reset shadow
    p.drawingContext.shadowBlur = 0;
    
    // Draw prediction info box
    p.fill(0, 0, 0, 180);
    p.stroke(255, 255, 0);
    p.strokeWeight(2);
    p.rect(p.width - 280, p.height - 100, 270, 85, 8);
    
    p.fill(255);
    p.textAlign(p.LEFT, p.TOP);
    p.textSize(16);
    p.text(`PREDICTED: ${predictedClass}`, p.width - 270, p.height - 90);
    
    if (trueClass !== null) {
        p.textSize(14);
        p.fill(trueClass === predictedClass ? p.color(0, 255, 0) : p.color(255, 100, 100));
        p.text(`TRUE: ${trueClass}`, p.width - 270, p.height - 65);
        
        // Show correct/incorrect indicator
        if (trueClass === predictedClass) {
            p.fill(0, 255, 0);
            p.text("✅ CORRECT!", p.width - 270, p.height - 40);
        } else {
            p.fill(255, 100, 100);
            p.text("❌ WRONG", p.width - 270, p.height - 40);
        }
    }
    
    p.textAlign(p.CENTER, p.CENTER);
    
    // Draw layer labels
    p.fill(255);
    p.textSize(14);
    p.text("Output Layer", 700, 30);
    p.text("10 Classes", 700, 50);
}
    
    function drawConnections(p) {
        if (!networkState.weights) return;
        
        p.strokeWeight(2);
        
        // Draw W1 connections (Input → Hidden)
        if (networkState.weights.W1) {
            for (let hiddenIdx = 0; hiddenIdx < networkState.weights.W1.length; hiddenIdx++) {
                for (let inputIdx = 0; inputIdx < networkState.weights.W1[hiddenIdx].length; inputIdx++) {
                    let weight = networkState.weights.W1[hiddenIdx][inputIdx];
                    
                    // Map weight to color: red (negative) → gray (zero) → blue (positive)
                    let alpha = p.map(Math.abs(weight), 0, 0.5, 30, 200);
                    if (weight > 0) {
                        p.stroke(0, 150, 255, alpha);
                    } else {
                        p.stroke(255, 50, 50, alpha);
                    }
                    
                    // Thickness based on magnitude
                    let thickness = p.map(Math.abs(weight), 0, 0.5, 0.5, 3);
                    p.strokeWeight(thickness);
                    
                    let x1 = 100;
                    let y1 = p.map(inputIdx, 0, networkState.weights.W1[hiddenIdx].length, 80, p.height - 80);
                    let x2 = 400;
                    let y2 = p.map(hiddenIdx, 0, networkState.weights.W1.length, 100, p.height - 100);
                    
                    p.line(x1, y1, x2, y2);
                }
            }
        }
        
        // Draw W2 connections (Hidden → Output)
        if (networkState.weights.W2) {
            for (let outputIdx = 0; outputIdx < networkState.weights.W2.length; outputIdx++) {
                for (let hiddenIdx = 0; hiddenIdx < networkState.weights.W2[outputIdx].length; hiddenIdx++) {
                    let weight = networkState.weights.W2[outputIdx][hiddenIdx];
                    
                    let alpha = p.map(Math.abs(weight), 0, 0.5, 30, 200);
                    if (weight > 0) {
                        p.stroke(0, 150, 255, alpha);
                    } else {
                        p.stroke(255, 50, 50, alpha);
                    }
                    
                    let thickness = p.map(Math.abs(weight), 0, 0.5, 0.5, 3);
                    p.strokeWeight(thickness);
                    
                    let x1 = 400;
                    let y1 = p.map(hiddenIdx, 0, networkState.weights.W2[outputIdx].length, 100, p.height - 100);
                    let x2 = 700;
                    let y2 = p.map(outputIdx, 0, networkState.weights.W2.length, 150, p.height - 150);
                    
                    p.line(x1, y1, x2, y2);
                }
            }
        }
        
        p.noFill();
    }
    
    function drawPhaseAnimation(p) {
        if (!networkState) return;
        
        let phase = networkState.phase;
        let progress = animationProgress;
        
        p.textSize(18);
        p.fill(255, 255, 0);
        
        if (phase === 'forward') {
            let x = p.lerp(150, 350, progress);
            p.fill(0, 255, 0, 200);
            p.circle(x, p.height/2, 20);
            p.fill(255, 255, 0);
            p.text("Forward Propagation →", p.width/2, p.height - 20);
        } else if (phase === 'backward') {
            let x = p.lerp(650, 450, progress);
            p.fill(255, 0, 0, 200);
            p.circle(x, p.height/2, 20);
            p.fill(255, 100, 100);
            p.text("← Backward Propagation", p.width/2, p.height - 20);
        } else if (phase === 'update') {
            p.fill(255, 200, 0);
            p.text("⚡ Updating Weights...", p.width/2, p.height - 20);
        }
    }
    
    function drawInfoBox(p) {
        // Draw info box in top-right
        p.fill(0, 0, 0, 150);
        p.stroke(255);
        p.strokeWeight(1);
        p.rect(p.width - 200, 10, 190, 80, 5);
        
        p.fill(255);
        p.textAlign(p.LEFT, p.TOP);
        p.textSize(12);
        p.text(`Iteration: ${networkState.iteration}`, p.width - 190, 20);
        p.text(`Loss: ${networkState.loss.toFixed(4)}`, p.width - 190, 40);
        p.text(`Accuracy: ${(networkState.accuracy * 100).toFixed(1)}%`, p.width - 190, 60);
        
        p.textAlign(p.CENTER, p.CENTER);
    }
};

// ==========================================
// SOCKET.IO HANDLERS
// ==========================================

function connectSocket() {
    socket = io.connect('http://' + document.domain + ':' + location.port);
    console.log("🔗 Connecting to server...");
    
    socket.on('connect', function() {
        console.log("✅ Connected to server");
    });
    
    socket.on('network_update', function(data) {
        console.log("📥 Received network update:", data.phase, "Iteration:", data.iteration);
        console.log("   Loss:", data.loss.toFixed(4), "Accuracy:", data.accuracy.toFixed(4));
        
        // Store the state for p5.js to render
        networkState = data;
        
        // Update UI metrics
        updateMetrics(data);
    });
    
    socket.on('validation_metrics', function(data) {
        console.log("📊 Validation metrics:", data);
        // You can extend this to plot a chart
    });
    
    socket.on('disconnect', function() {
        console.log("❌ Disconnected from server");
    });
}

function updateMetrics(data) {
    // Update metric boxes
    document.getElementById('iteration').textContent = data.iteration;
    document.getElementById('loss').textContent = data.loss.toFixed(4);
    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('phase').textContent = data.phase.toUpperCase();
}

// ==========================================
// CONTROLS
// ==========================================

document.getElementById('startBtn').addEventListener('click', function() {
    console.log("🚀 Starting training...");
    socket.emit('start_training');
    this.disabled = true;
    document.getElementById('pauseBtn').disabled = false;
});

document.getElementById('pauseBtn').addEventListener('click', function() {
    isPaused = !isPaused;
    this.textContent = isPaused ? 'Resume' : 'Pause';
    console.log(isPaused ? "⏸️ Paused" : "▶️ Resumed");
});

// ==========================================
// INITIALIZATION
// ==========================================

// Initialize p5.js
let networkViz = new p5(sketch);

// Initialize socket when page loads
window.onload = function() {
    console.log("🌟 Page loaded, initializing...");
    connectSocket();
};