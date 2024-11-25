import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.weights1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.bias2 = np.zeros((1, output_dim))

        
    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError("Unsupported activation function")


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Output layer uses sigmoid

        # Debug: Print activations
        print(f"Hidden activations (a1): {self.a1}")

        # For visualization
        self.hidden_features = self.a1
        self.output = self.a2

        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        # TODO: compute gradients using chain rule
        dz2 = self.a2 - y                       
        dw2 = np.dot(self.a1.T, dz2) / m        
        db2 = np.sum(dz2, axis=0, keepdims=True) / m 

        dz1 = np.dot(dz2, self.weights2.T) * self.activation_derivative(self.z1) 
        dw1 = np.dot(X.T, dz1) / m       
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  

        # TODO: update weights with gradient descent
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2
        # TODO: store gradients for visualization
        self.gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }
        pass

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize input data
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    print(f"Processing frame: {frame}")
    try:
        ax_hidden.clear()
        ax_input.clear()
        ax_gradient.clear()

        # Perform training steps
        for _ in range(10):
            mlp.forward(X)
            mlp.backward(X, y)

        # Plot hidden features
        hidden_features = mlp.hidden_features
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Features")
        ax_hidden.set_xlabel("Neuron 1 Activation")
        ax_hidden.set_ylabel("Neuron 2 Activation")
        ax_hidden.set_zlabel("Neuron 3 Activation")

        # Plot decision boundary in input space
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = mlp.forward(grid).reshape(xx.shape)
        ax_input.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="bwr", alpha=0.3)
        ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")
        ax_input.set_title("Input Space Decision Boundary")
        ax_input.set_xlabel("Feature 1")
        ax_input.set_ylabel("Feature 2")

        # Gradient visualization
        # Compute gradient magnitudes
        gradients = np.sqrt(np.sum(mlp.gradients["dw1"]**2, axis=0))  # Gradient magnitudes

        # Normalize gradients for visualization
        gradients_scaled = gradients / (np.max(gradients) + 1e-6)  # Avoid divide-by-zero

        # Visualize scaled gradients
        for i, grad in enumerate(gradients_scaled):
            circle = Circle((i, 0), grad * 5, alpha=0.5, color="blue")  # Amplify the size
            ax_gradient.add_patch(circle)

        ax_gradient.set_xlim(-1, mlp.weights1.shape[1])
        ax_gradient.set_ylim(-1, 1)
        ax_gradient.set_title("Gradient Visualization")
    except Exception as e:
        print(f"Error in update() during frame {frame}: {e}")
        raise

def visualize(activation, lr, step_num):
    try:
        # Generate data
        X, y = generate_data()

        # Initialize the MLP
        mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

        # Set up visualization
        matplotlib.use('agg')  # Use a non-interactive backend
        fig = plt.figure(figsize=(21, 7))
        ax_hidden = fig.add_subplot(131, projection='3d')
        ax_input = fig.add_subplot(132)
        ax_gradient = fig.add_subplot(133)

        # Convert frames to a list to avoid generator exhaustion
        frames = list(range(step_num // 10))

        # Create the animation
        ani = FuncAnimation(
            fig,
            partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
            frames=frames,
            repeat=False
        )

        # Save the animation as a GIF
        output_path = os.path.join("results", "visualize.gif")
        ani.save(output_path, writer='pillow', fps=10)
        plt.close()
        print(f"Animation saved successfully to {output_path}")
    except Exception as e:
        print(f"Error in visualize(): {e}")
        raise

if __name__ == "__main__":
    # Allow the user to select the activation function
    print("Choose an activation function from the following options: relu, tanh, sigmoid")
    activation = input("Enter activation function: ").strip().lower()

    # Validate the input
    if activation not in ["relu", "tanh", "sigmoid"]:
        print("Invalid activation function. Please choose from relu, tanh, sigmoid.")
        exit()

    # Set learning rate and step number
    lr = 0.1
    step_num = 1000

    print(f"Using activation function: {activation}")
    visualize(activation, lr, step_num)
