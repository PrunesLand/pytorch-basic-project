# PyTorch Iris Classification

This project is a simple introduction to PyTorch, demonstrating how to build a neural network to classify the famous Iris dataset.

## Purpose

The goal of this project is to provide a clear and concise example of a complete machine learning pipeline using PyTorch. It's intended for beginners who are just starting with PyTorch and want to see a practical example of how to:
- Load and preprocess data.
- Define a neural network.
- Train the network.
- Evaluate its performance.
- Make predictions on new data.

## Setup

To run this project, you'll need Python and the libraries listed in `requirements.txt`.

1.  **Clone the repository (or download the files).**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Once you have set up the environment, you can run the main script:

```bash
python main.py
```

You should see output that shows the training loss at different epochs, the final accuracy on the test set, and an example prediction.

## PyTorch Concepts Used

This project introduces several fundamental PyTorch concepts:

*   **Tensors:** The primary data structure in PyTorch. They are similar to NumPy arrays but can also be used on a GPU to accelerate computing. In this project, we convert our data from NumPy arrays into PyTorch Tensors.

*   **`torch.nn.Module`:** The base class for all neural network modules. Our `IrisNet` class inherits from `nn.Module`. This gives us access to a lot of useful functionality for building and training models.

*   **`nn.Linear`:** A linear transformation (`y = Wx + b`). This is a standard building block for neural networks, also known as a fully-connected or dense layer. Our network has two linear layers.

*   **Activation Functions (`torch.relu`):** These functions introduce non-linearity into the model, allowing it to learn more complex patterns. We use the Rectified Linear Unit (ReLU) activation function after our first linear layer.

*   **Loss Function (`nn.CrossEntropyLoss`):** This measures how well the model is performing. For multi-class classification problems like this one, `CrossEntropyLoss` is a common choice. It combines `nn.LogSoftmax` and `nn.NLLLoss` in one single class.

*   **Optimizer (`torch.optim.Adam`):** An algorithm that adjusts the model's parameters (weights and biases) during training to minimize the loss. Adam is a popular and effective optimization algorithm.

*   **Training Loop:** The core of the training process. In each epoch (one full pass through the training data), we:
    1.  Reset the gradients (`optimizer.zero_grad()`).
    2.  Perform a forward pass to get predictions (`model(X_train)`).
    3.  Calculate the loss (`criterion(outputs, y_train)`).
    4.  Perform a backward pass to compute gradients (`loss.backward()`).
    5.  Update the model's parameters (`optimizer.step()`).

*   **`torch.no_grad()`:** A context manager that disables gradient calculation. This is important for evaluation and inference because it reduces memory consumption and speeds up computations when we don't need to update the model.