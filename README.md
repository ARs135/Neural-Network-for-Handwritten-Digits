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
Make sure you have Python [Download Python](https://python.org/downloads)
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
When you run the script, if pre-trained weights are found, you'll be given the option to continue training or launch the digit recognizer. Once in the GUI, draw digits using your mouse. Click **"Predict"** to see the network's guess, or **"Clear"** to try again.

## 🧪 Pre-trained Weights
This repository includes pre-trained weights [model_weights.npz](model_weights.npz) so you can run the recognizer immediately without training from scratch. But if you want to re-train it, just delete the weights file and run the script — it will start training automatically.

## 📚 Dataset
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — 70,000 labeled handwritten digits

## 🧮 Miscellaneous
- **Input layer**: 784 neurons (28×28 pixels)
- **Hidden layers**: 128 → 64 → 32 neurons
- **Output layer**: 10 neurons (digits 0–9)

## 📝 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## ✍️ Author
Made by [ARs135](https://github.com/ARs135)
