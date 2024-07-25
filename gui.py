import streamlit as st
import random
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def perceptron_quality(p_perceptron: Perceptron, p_num_tests: int, p_logic_to_apply: str) -> float:
    """
    Test the quality of the perceptron by checking its accuracy.
    """
    num_correct = 0
    all_possible_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _ in range(p_num_tests):
        input_data = random.choice(all_possible_inputs)
        predicted_output = p_perceptron.predict(input_data)

        # Calculate expected output based on the logic being trained
        if p_logic_to_apply == 'AND':
            p_expected_output = 1 if all(input_data) else 0
        elif p_logic_to_apply == 'OR':
            p_expected_output = 1 if any(input_data) else 0
        else:
            raise ValueError("Invalid logic_to_apply")

        if predicted_output == p_expected_output:
            num_correct += 1
    return (num_correct / p_num_tests) * 100


def plot_decision_boundary(pctron: Perceptron, range: Tuple[int] = (-1, 2)):
    """Plots the decision boundary of the perceptron and returns the figure."""
    fig, ax = plt.subplots()  # Create a figure and axes

    weights = pctron.weights
    w1, w2, bias = weights[1], weights[2], weights[0]

    # Calculate x and y values for the decision boundary line
    x = np.linspace(range[0], range[1], 100)
    y = (-w1 * x - bias) / w2

    # Plot the decision boundary
    ax.plot(x, y, linestyle='--', color='grey', label='Decision Boundary')
    ax.legend(loc='upper right')

    return fig


def plot_errors(p_errors, p_epochs, p_legend):
    """Plots the error curve and associated information."""
    fig, ax = plt.subplots()
    ax.plot(range(1, p_epochs + 1), p_errors, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Error')
    ax.set_title('Perceptron Training')

    # Add text box with information
    if p_legend:
        props = dict(boxstyle='round', facecolor='tab:orange', alpha=0.6)
        ax.text(0.95, 0.95, p_legend, transform=ax.transAxes, fontsize=10,
                horizontalalignment='right', verticalalignment='top', bbox=props)

    return fig  # Return the figure for Streamlit to display


def create_legend(*args):
    """
    Create a legend to display the parameters
    used in training and display on the training plot.
    """
    info_text = f"Epochs: {args[0]}" + "\n"
    info_text += f"Logic: {args[1]}" + "\n"
    info_text += f"Learning rate: {args[2]}" + "\n"
    info_text += f"Test count: {args[3]}" + "\n"
    info_text += f"Activator: {args[4]}" + "\n"
    info_text += f"Use Threshold: {args[5]}"
    return info_text


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

# Plotting the errors
error_fig = plot_errors(errors, epochs, legend)
st.pyplot(error_fig)

# Plotting the decision boundary
boundary_fig = plot_decision_boundary(perceptron)

# Plot the input data points (on the boundary figure) and apply styling
for x, expected_output in zip(perceptron.inputs, perceptron.expected_output):
    color = 'blue' if expected_output == 0 else 'red'
    boundary_fig.gca().scatter(x[0], x[1], c=color, label=f'Class {expected_output}')

# Style the legend for consistency
legend_elements = boundary_fig.gca().get_legend_handles_labels()  # Get the legend elements from the plot
boundary_fig.gca().legend(*legend_elements, loc='upper right', fontsize=10, facecolor='tab:orange', framealpha=0.6, edgecolor='black')
st.pyplot(boundary_fig)

st.write(f"Accuracy: {acc:.2f}%")

