import numpy as np
import tkinter as tk
from tensorflow.keras.datasets import mnist

# --- Activation functions ---
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# --- Loss + utility ---
def cross_entropy(pred, y): return -np.mean(np.sum(y * np.log(pred + 1e-9), axis=1))
def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

# --- Neural Network with 3 hidden layers ---
class NeuralNet:
    def __init__(self):
        self.w1 = np.random.randn(784, 256) * np.sqrt(2 / 784)
        self.b1 = np.zeros((1, 256))

        self.w2 = np.random.randn(256, 128) * np.sqrt(2 / 256)
        self.b2 = np.zeros((1, 128))

        self.w3 = np.random.randn(128, 64) * np.sqrt(2 / 128)
        self.b3 = np.zeros((1, 64))

        self.w4 = np.random.randn(64, 10) * np.sqrt(2 / 64)
        self.b4 = np.zeros((1, 10))

    def forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = relu(self.z3)

        self.z4 = self.a3 @ self.w4 + self.b4
        self.a4 = softmax(self.z4)
        return self.a4

    def backward(self, x, y, out, lr):
        m = x.shape[0]

        dz4 = out - y
        dw4 = self.a3.T @ dz4 / m
        db4 = np.sum(dz4, axis=0, keepdims=True) / m

        dz3 = dz4 @ self.w4.T * relu_derivative(self.z3)
        dw3 = self.a2.T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = dz3 @ self.w3.T * relu_derivative(self.z2)
        dw2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * relu_derivative(self.z1)
        dw1 = x.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.w4 -= lr * dw4
        self.b4 -= lr * db4
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

    def train(self, x, y, epochs=150, lr=0.05, save_every=50, batch_size=64, lr_decay=0.95):
        num_samples = x.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                out = self.forward(x_batch)
                self.backward(x_batch, y_batch, out, lr)

            # Decay learning rate after each epoch
            lr *= lr_decay

            # Track loss on full dataset every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                full_out = self.forward(x)
                loss = cross_entropy(full_out, y)
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, LR: {lr:.5f}")

            if (epoch + 1) % save_every == 0:
                self.save_weights(f"model_weights_epoch{epoch + 1}.npz")
                print(f"Saved weights at epoch {epoch + 1}")

    def predict(self, x):
        return self.forward(x)

    def save_weights(self, filename="model_weights.npz"):
        np.savez(filename,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3,
                 w4=self.w4, b4=self.b4)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename="model_weights.npz"):
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.w3 = data['w3']
        self.b3 = data['b3']
        self.w4 = data['w4']
        self.b4 = data['b4']
        print(f"Weights loaded from {filename}")

# --- Load MNIST data ---
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
y_train_onehot = one_hot(y_train)

# --- Train or load model ---
# --- Choose mode ---
model = NeuralNet()
mode = input("Choose mode:\n1. Train model\n2. Run recognizer (UI only)\n> ")

if mode == "1":
    try:
        model.load_weights("model_weights.npz")
        print("Continuing training from loaded weights...")
    except FileNotFoundError:
        print("No saved weights found, training model from scratch...")

    model.train(x_train, y_train_onehot, epochs=150, lr=0.05, batch_size=64, lr_decay=0.97)
    model.save_weights("model_weights.npz")

elif mode == "2":
    try:
        model.load_weights("model_weights.npz")
    except FileNotFoundError:
        print("No saved weights found. You must train the model first.")
        exit()
else:
    print("Invalid choice.")
    exit()

# --- Tkinter UI: Pixel Drawing ---
cell_size = 10
grid_size = 28

def draw_cell(event):
    col = event.x // cell_size
    row = event.y // cell_size

    brush = [
        [0.1, 0.3, 0.1],
        [0.3, 1.0, 0.3],
        [0.1, 0.3, 0.1]
    ]

    for i in range(-1, 2):
        for j in range(-1, 2):
            r, c = row + i, col + j
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r][c] = min(1.0, grid[r][c] + brush[i + 1][j + 1])
                brightness = int(grid[r][c] * 255)
                color = f"#{brightness:02x}{brightness:02x}{brightness:02x}"
                canvas.create_rectangle(
                    c * cell_size, r * cell_size,
                    (c + 1) * cell_size, (r + 1) * cell_size,
                    fill=color, outline="gray"
                )

def clear():
    global grid
    grid = np.zeros((grid_size, grid_size))
    canvas.delete("all")
    draw_grid()

def draw_grid():
    for i in range(grid_size):
        for j in range(grid_size):
            canvas.create_rectangle(
                j * cell_size, i * cell_size,
                (j + 1) * cell_size, (i + 1) * cell_size,
                fill="black", outline="gray"
            )

def predict_and_show():
    data = grid.reshape(1, 784)
    preds = model.predict(data).flatten()
    most_likely = np.argmax(preds)

    result.delete('1.0', tk.END)
    for i, prob in enumerate(preds):
        tag = " (Most Likely)" if i == most_likely else ""
        result.insert(tk.END, f"{i}: {prob:.4f}{tag}\n")

# --- Setup GUI ---
root = tk.Tk()
root.title("Pixel Digit Recognizer")

grid = np.zeros((grid_size, grid_size))

canvas = tk.Canvas(root, width=grid_size * cell_size, height=grid_size * cell_size, bg='black')
canvas.pack()
canvas.bind("<B1-Motion>", draw_cell)
canvas.bind("<Button-1>", draw_cell)

tk.Button(root, text="Predict", command=predict_and_show).pack()
tk.Button(root, text="Clear", command=clear).pack()

# ⬇⬇⬇ Here's the updated line
result = tk.Text(root, height=12, width=30)
result.pack()

draw_grid()
root.mainloop()
