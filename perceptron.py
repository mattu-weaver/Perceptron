from typing import Dict, List, Union, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self: 'Perceptron',
                 logic_to_apply: str = 'AND',
                 input_size: int = 2,
                 learning_rate: float = 0.002,
                 activator: str = 'SIGMOID',
                 thresholds: Dict[str, float] = None,
                 use_threshold=True):

        self.weights = np.random.randn(input_size + 1)
        self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.learning_rate = learning_rate
        self.activator = activator
        self.input_size = input_size
        self.output = None
        self.logic_to_apply = logic_to_apply.upper()
        self.expected_and_output = [0, 0, 0, 1]
        self.expected_or_output = [0, 1, 1, 1]
        self.expected_xor_output = [0, 1, 1, 0]
        self.expected_output = self.get_expected_output()
        self.use_threshold = use_threshold

        self.thresholds = thresholds if thresholds is not None else {
            "step": 0,
            "sigmoid": 0.5,
            "tanh": 0,
            "relu": 0,
        }

    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)  # 1 if x >= 0, else 0

    @staticmethod
    def sigmoid(x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def relu(x: np.ndarray):
        return np.maximum(0, x)

    def get_activation_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Retrieves the activation function with or without its threshold."""
        activation_name = self.activator.lower()
        activation_fn = getattr(self, activation_name, self.step)

        if self.use_threshold:
            threshold = self.thresholds[activation_name]
            return lambda x: np.where(activation_fn(x) >= threshold, 1, 0)  # Apply threshold
        else:
            return activation_fn  # No thresholdhin lambda

    def get_expected_output(self: 'Perceptron') -> Union[List[int], None]:
        if self.logic_to_apply == 'AND':
            return self.expected_and_output
        elif self.logic_to_apply == 'OR':
            return self.expected_or_output
        elif self.logic_to_apply == 'XOR':
            return self.expected_xor_output
        else:
            return None

    def train(self, epochs: int = 10) -> None:
        activation_function = getattr(self, self.activator.lower(), self.step)
        errors = []  # List to store errors for each epoch

        for epoch in range(epochs):
            epoch_error = 0  # Error for the current epoch
            for x, expected_output in zip(self.inputs, self.expected_output):
                x = np.insert(x, 0, 1)
                weighted_sum = np.dot(x, self.weights)

                predicted_output = int(activation_function(weighted_sum))
                error = expected_output - predicted_output
                epoch_error += abs(error)  # Accumulate error for the epoch
                self.weights += self.learning_rate * error * x

            errors.append(epoch_error)  # Store the epoch's total error

        # Plot the error over epochs
        plt.plot(range(1, epochs + 1), errors, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Total Error')
        plt.title('Error vs. Epoch')
        plt.show()

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.insert(x, 0, 1)
        weighted_sum = np.dot(x, self.weights)

        # Get the activation function with threshold
        activation_function = self.get_activation_function()
        activation = activation_function(weighted_sum)  # Calculate activation

        # Thresholding is done inside the get_activation_function method
        predicted_output = activation
        return predicted_output
