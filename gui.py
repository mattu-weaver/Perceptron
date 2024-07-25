import streamlit as st
import random
from typing import List
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

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
        # Adjusted fontsize
        legend_text = ax.text(0.95, 0.95, legend, transform=ax.transAxes, fontsize=8,
                             horizontalalignment='right', verticalalignment='top', bbox=props)
        legend_text._get_wrap_line_width = lambda : legend_text.get_window_extent().width  # Adjust line wrapping for smaller box

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
    ax = fig.gca()
    handles = []
    labels = []
    for x, expected_output in zip(pctron.inputs, pctron.expected_output):
        color = 'blue' if expected_output == 0 else 'red'
        scatter = ax.scatter(x[0], x[1], c=color, label=f'Class {expected_output}')
        handles.append(scatter)
        labels.append(f'Class {expected_output}')

    # Manually create the legend with desired handles and labels
    leg = ax.legend(handles, labels, loc='upper right', fontsize=8, facecolor='tab:orange', framealpha=0.6,
                    edgecolor='black')

    # Add empty space after 'Decision Boundary'
    ax.legend(handles=[plt.Line2D([0], [0], color='grey', linestyle='--')], labels=['\nDecision Boundary'],
              loc='upper right', fontsize=8, facecolor='tab:orange', framealpha=0.6, edgecolor='black')

    # Adjust line wrapping for smaller box
    for text in leg.get_texts():
        text._get_wrap_line_width = lambda: text.get_window_extent().width

    return fig


def plot_weight_changes(weight_history: List[np.ndarray], epochs: int):
    """Plots the change in weights over epochs."""
    fig, ax = plt.subplots()
    num_weights = len(weight_history[0])
    weight_labels = [f"Weight {i}" for i in range(num_weights)]

    # Extract weights for each epoch
    for i in range(num_weights):
        weight_values = [weights[i] for weights in weight_history]
        ax.plot(range(1, epochs + 1), weight_values, label=weight_labels[i], marker='o')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.set_title('Weight Changes Over Epochs')

    # Create the legend
    legend = ax.legend(loc='upper right', fontsize=8, facecolor='tab:orange', framealpha=0.6, edgecolor='black')

    # Add padding to legend labels
    for text in legend.get_texts():
        text.set_position((text.get_position()[0], text.get_position()[1] - 0.2))  # Adjust vertical position for padding

    return fig


def plot_weighted_sums(weighted_sums, logic_to_apply):
    """Plots a histogram of weighted sums."""
    fig, ax = plt.subplots()
    ax.hist(weighted_sums, bins=10, edgecolor='k', alpha=0.7) # Adjust bins as needed
    ax.set_xlabel('Weighted Sum')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Weighted Sums for {logic_to_apply} Logic')
    return fig


# Streamlit UI
with st.sidebar:
    st.header("Configuration")
    epochs = st.slider('Epochs', 100, 1000, 200)
    num_tests = st.sidebar.slider('Test Count', 100, 1000, 500)
    learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.1, 0.01)
    activator = st.selectbox('Activator', ['STEP', 'SIGMOID', 'TANH', 'RELU'], index=0)
    use_threshold = st.checkbox('Use Threshold')
    logic_to_apply = st.selectbox('Logic to Apply', ['AND', 'OR', 'XOR'], index=0)

legend = create_legend(epochs, logic_to_apply, learning_rate, num_tests, activator, use_threshold)

perceptron = Perceptron(logic_to_apply=logic_to_apply, learning_rate=learning_rate,
                        activator=activator, use_threshold=use_threshold)

weights, errors, weight_history, weighted_sums = perceptron.train(epochs)  # Capture weighted_sums
acc = perceptron_quality(perceptron, num_tests, logic_to_apply)

col1, col2 = st.columns(2)

with col1:
    error_fig = plot_errors(errors, epochs, legend)
    st.pyplot(error_fig)  # Display the error plot

with col2:
    boundary_fig = plot_decision_boundary(perceptron)
    boundary_fig = plot_data_points(perceptron, boundary_fig)
    st.pyplot(boundary_fig)

col3, col4 = st.columns(2)

with col3:
    weights_fig = plot_weight_changes(weight_history, epochs)
    st.pyplot(weights_fig)
with col4:
    weighted_sums_fig = plot_weighted_sums(weighted_sums, logic_to_apply)
    st.pyplot(weighted_sums_fig)

st.write(f"Accuracy: {acc:.2f}%")
