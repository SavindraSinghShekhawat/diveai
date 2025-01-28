import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from diveai.visualization import PlotBuilder

def gradient_descent(X, y, initial_weights, initial_bias, learning_rate, iterations, dive=False):
    """
    Perform gradient descent with optional interactive visualization and logging.

    Args:
        X (numpy.ndarray): Feature matrix of shape (m, n).
        y (numpy.ndarray): Target vector of shape (m,).
        initial_weights (numpy.ndarray): Initial weights of shape (n, 1).
        initial_bias (float): Initial bias term.
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations.
        dive (bool): If True, display plots and print progress. If False, run silently.

    Returns:
        tuple: Optimized weights and bias.
    """
    # Initialize weights and bias
    weights = initial_weights
    bias = initial_bias

    # Logs for plotting
    weight_log = []
    bias_log = []
    cost_log = []

    m = X.shape[0]  # Number of examples

    if dive:
        # Create a Plotly Figure with 3 subplots:
        pb = PlotBuilder(rows=1, cols=3, title="Gradient Descent Process", subplot_titles=("Cost vs Iterations", "Weights and Bias vs Iterations", "Data & Fit Line"))

        pb.add_plot([], [], row=0, col=0, plot_type="line", color="blue", label="Cost")
        pb.add_plot([], [], row=0, col=1, plot_type="line", color="orange", label="Weights")
        pb.add_plot([], [], row=0, col=1, plot_type="line", color="green", label="Bias")
        pb.add_plot(X[:, 0], y if y.ndim == 1 else y[:, 0], row=0, col=2, plot_type="scatter", color="black", label="Data")
        pb.add_plot([], [], row=0, col=2, plot_type="line", color="red", label="Fit Line")

        pb.set_labels(row=0, col=0, x_label="Iterations", y_label="Cost (MSE)")
        pb.set_labels(row=0, col=1, x_label="Iterations", y_label="Value")
        pb.set_labels(row=0, col=2, x_label="X", y_label="y")

        pb.show()

    # Gradient descent loop
    for i in range(iterations):
        # Compute predictions
        y_pred = np.dot(X, weights) + bias

        # Compute gradients
        dw = -(1 / m) * np.dot(X.T, (y - y_pred))
        db = -(1 / m) * np.sum(y - y_pred)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Compute cost (Mean Squared Error)
        cost = np.mean((y - y_pred) ** 2)

        # Log values for plotting
        weight_log.append(weights[0][0])  # Assuming weights is shape (1,1)
        bias_log.append(bias)
        cost_log.append(cost)

        if dive:
            pb.update_trace(list(range(len(cost_log))), cost_log, row=0, col=0, trace=0)
            pb.update_trace(list(range(len(weight_log))), weight_log, row=0, col=1, trace=1)
            pb.update_trace(list(range(len(bias_log))), bias_log, row=0, col=1, trace=2)
            pb.update_trace(X[:, 0], y_pred if y_pred.ndim == 1 else y_pred[:, 0], row=0, col=2, trace=4, auto_range=True)
            
            # Print progress (optional)
            # print(
            #     f"Iteration {i+1}/{iterations}, Cost: {cost:.6f}, Weight: {weights[0][0]:.6f}, Bias: {bias:.6f}"
            # )

    if dive:
        print("Gradient Descent Complete!")

    return weights, bias