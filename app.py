from perceptron import Perceptron
import numpy as np
import toml
import seaborn as sns
import matplotlib.pyplot as plt


def perceptron_quality(num_tests):
    num_correct = 0

    for _ in range(num_tests):
        input_data = np.random.randint(0, 2, size=2)

        # Convert NumPy array to a list of regular Python integers
        input_data_list = input_data.tolist()

        predicted_output = perceptron.predict(input_data)

        expected_output = perceptron.get_expected_output()[
            perceptron.inputs.index(input_data_list)  # Use the converted list
        ]

        if predicted_output == expected_output:
            num_correct += 1

    return (num_correct / num_tests) * 100


cfg = toml.load('.streamlit/config.toml')
perceptron = Perceptron(logic_to_apply='OR', learning_rate=0.002,activator='SIGMOID', use_threshold=False)
perceptron.train(100)

acc = perceptron_quality(500)
print(f"Accuracy: {acc:.2f}%")


def train_and_track_error(perceptron, epochs):
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for x, expected_output in zip(perceptron.inputs, perceptron.expected_output):
            predicted_output = perceptron.predict(x)
            error = expected_output - predicted_output
            x_with_bias = np.insert(x, 0, 1)  # Add bias only for weight update
            perceptron.weights += perceptron.learning_rate * error * x_with_bias
            total_error += abs(error)
        errors.append(total_error)
    return errors


# Training with error tracking
errors = train_and_track_error(perceptron, 1000)

# Plotting with Seaborn
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, 1001), y=errors, marker="o")
plt.title("Perceptron Training Error vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Total Absolute Error")
plt.show()