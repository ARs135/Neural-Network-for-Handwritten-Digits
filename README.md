# 🧠 Neural Network for Handwritten Digits

This is a simple but powerful neural network built from scratch in Python (using only NumPy) to recognize handwritten digits. It uses the MNIST dataset for training and includes a **Tkinter GUI** to draw and classify digits live.

## 🔧 Features

- Fully connected neural network with 3 hidden layers
- ReLU activation and softmax output
- Cross-entropy loss function
- Custom training loop with batch updates and learning rate decay
- Saves and loads pre-trained weights (`.npz` files)
- Interactive drawing canvas (Tkinter GUI)
- Predicts drawn digits live

## 📦 Requirements

- `numpy`
- `tensorflow` (for loading MNIST)
- `tkinter` (usually comes with Python)

Install dependencies with:

```bash
pip install numpy tensorflow
```
## 🚀 Running the App
```bash
python main.py
```
A window will open where you can draw digits using your mouse. Click **"Predict"** to see what the network thinks it is. You can also click **"Clear"** to try again.

## 🧪 Pre-trained Weights
This repository includes pre-trained weights (`model_weights.npz`) so you can run the recognizer immediately without training from scratch. But if you want to re-train it, just delete the weights file and run the script — it will start training automatically.

## 📚 Dataset
+ [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — 70,000 labeled handwritten digits

---

# Made by ARs135
