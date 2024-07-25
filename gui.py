import streamlit as st
import random
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np


def perceptron_quality(pctron: Perceptron, test_count: int, logic: str) -> float:
    """Test the quality of the perceptron by checking its accuracy."""
    num_correct = 0
    all_possible_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _ in range(test_count):
        input_data = random.choice(all_possible_inputs)
        predicted_output = pctron.predict(input_data)

        if logic == 'AND':
            expected_output = 1 if all(input_data) else 0
        elif logic == 'OR':
            expected_output = 1 if any(input_data) else 0
        elif logic == 'XOR':
            expected_output = 1 if sum(input_data) % 2 == 1 else 0
        else:
            raise ValueError("Invalid logic_to_apply")

        if predicted_output == expected_output:
            num_correct += 1
    return (num_correct / test_count) * 100


def create_legend(*args):
    """Create a legend to display parameters used in training."""
    info_text = f"Epochs: {args[0]}\n"
    info_text += f"Logic: {args[1]}\n"
    info_text += f"Learning rate: {args[2]}\n"
    info_text += f"Test count: {args[3]}\n"
    info_text += f"Activator: {args[4]}\n"
    info_text += f"Use Threshold: {args[5]}"
    return info_text


def plot_errors(errors, epochs, legend):
    """Plots the error curve and associated information."""
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), errors, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Error')
    ax.set_title('Perceptron Training')

    if legend:
        props = dict(boxstyle='round', facecolor='tab:orange', alpha=0.6)
        ax.text(0.95, 0.95, legend, transform=ax.transAxes, fontsize=10,
                horizontalalignment='right', verticalalignment='top', bbox=props)
    return fig


def plot_decision_boundary(pctron: Perceptron, x_min=-1, x_max=2, y_min=-1, y_max=2):
    """Plots the decision boundary only."""
    fig, ax = plt.subplots()
    weights = pctron.weights
    w1, w2, bias = weights[1], weights[2], weights[0]
    x = np.linspace(x_min, x_max, 100)
    y = (-w1 * x - bias) / w2
    ax.set_title('Decision Boundary')
    ax.plot(x, y, linestyle='--', color='grey', label='Decision Boundary')

    return fig


def plot_data_points(pctron, fig):
    """Plots the input data points on the figure."""
    ax = fig.gca()  # Get the current axis from the figure
    for x, expected_output in zip(pctron.inputs, pctron.expected_output):
        color = 'blue' if expected_output == 0 else 'red'
        ax.scatter(x[0], x[1], c=color, label=f'Class {expected_output}')
    ax.legend(loc='upper right', fontsize=10, facecolor='tab:orange', framealpha=0.6, edgecolor='black')
    return fig


# Streamlit UI
st.markdown('# A Simple Perceptron')
epochs = st.sidebar.slider('Epochs', 100, 1000, 200)
num_tests = st.sidebar.slider('Test Count', 100, 1000, 500)
learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.1, 0.01)
activator = st.sidebar.selectbox('Activator', ['STEP', 'SIGMOID', 'TANH', 'RELU'], index=0)
use_threshold = st.sidebar.checkbox('Use Threshold')
logic_to_apply = st.sidebar.selectbox('Logic to Apply', ['AND', 'OR', 'XOR'], index=0)

legend = create_legend(epochs, logic_to_apply, learning_rate, num_tests, activator, use_threshold)

perceptron = Perceptron(logic_to_apply=logic_to_apply, learning_rate=learning_rate,
                        activator=activator, use_threshold=use_threshold)

errors = perceptron.train(epochs)
acc = perceptron_quality(perceptron, num_tests, logic_to_apply)

# Plotting
error_fig = plot_errors(errors, epochs, legend)
st.pyplot(error_fig)  # Display the error plot

boundary_fig = plot_decision_boundary(perceptron)
boundary_fig = plot_data_points(perceptron, boundary_fig)
st.pyplot(boundary_fig)

st.write(f"Accuracy: {acc:.2f}%")
