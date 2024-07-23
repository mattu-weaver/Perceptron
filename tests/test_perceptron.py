import pytest
from perceptron import Perceptron
import numpy as np


@pytest.fixture
def ptron():
    return Perceptron()


@pytest.fixture
def perceptron_and():
    return Perceptron(logic_to_apply='AND')


@pytest.fixture
def perceptron_or():
    return Perceptron(logic_to_apply='OR')


@pytest.fixture
def perceptron_xor():
    return Perceptron(logic_to_apply='XOR')


def test_and_weight_updates(perceptron_and):  # Using fixture directly
    initial_weights = perceptron_and.weights.copy()
    perceptron_and.train(epochs=5)
    assert not np.array_equal(initial_weights, perceptron_and.weights), "Weights should have changed"


def test_or_weight_updates(perceptron_or):  # Using fixture directly
    initial_weights = perceptron_or.weights.copy()
    perceptron_or.train(epochs=5)
    assert not np.array_equal(initial_weights, perceptron_or.weights), "Weights should have changed"


def test_xor_weight_updates(perceptron_xor):  # Using fixture directly
    initial_weights = perceptron_xor.weights.copy()
    perceptron_xor.train(epochs=5)
    assert not np.array_equal(initial_weights, perceptron_xor.weights), "Weights should have changed"


def test_perceptron_config(ptron):
    assert ptron.inputs == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert ptron.learning_rate == 0.006
    assert ptron.activator == 'SIGMOID'
    assert ptron.input_size == 2


def test_perceptron_logic():
    p = Perceptron(logic_to_apply='AnD')
    assert p.get_expected_output() == [0, 0, 0, 1]
    p = Perceptron(logic_to_apply='or')
    assert p.get_expected_output() == [0, 1, 1, 1]
    p = Perceptron(logic_to_apply='XOR')
    assert p.get_expected_output() == [0, 1, 1, 0]


def test_step_positive_values():
    # Test positive input values
    x = np.array([1, 2.5, 10, 0.01])
    expected_output = np.array([1, 1, 1, 1])
    result = Perceptron.step(x)
    assert np.array_equal(result, expected_output)


def test_step_negative_values():
    # Test negative input values
    x = np.array([-1, -3.2, -100, -0.005])
    expected_output = np.array([0, 0, 0, 0])
    result = Perceptron.step(x)
    assert np.array_equal(result, expected_output)


def test_step_zero():
    # Test zero input
    x = np.array([0])
    expected_output = np.array([1])  # Step function should output 1 for zero
    result = Perceptron.step(x)
    assert np.array_equal(result, expected_output)


def test_step_mixed_values():
    # Test mixed positive and negative inputs
    x = np.array([-2, 0, 3.14, -1.5])
    expected_output = np.array([0, 1, 1, 0])
    result = Perceptron.step(x)
    assert np.array_equal(result, expected_output)


def test_step_empty_array():
    # Test empty input array
    x = np.array([])
    expected_output = np.array([])
    result = Perceptron.step(x)
    assert np.array_equal(result, expected_output)


# Sigmoid Tests
def test_sigmoid_range():
    # Test that sigmoid output is within (0, 1)
    x = np.linspace(-10, 10, 100)
    result = Perceptron.sigmoid(x)
    assert np.all((result > 0) & (result < 1))


def test_sigmoid_symmetry():
    # Test symmetry around 0
    x = np.array([-2, -1, 0, 1, 2])
    result = Perceptron.sigmoid(x)
    assert np.allclose(result, 1 - Perceptron.sigmoid(-x))  # Check if sigmoid(x) = 1 - sigmoid(-x)


# Tanh Tests
def test_tanh_range():
    # Test that tanh output is within (-1, 1)
    x = np.linspace(-10, 10, 100)
    result = Perceptron.tanh(x)
    assert np.all((result > -1) & (result < 1))


def test_tanh_symmetry():
    # Test symmetry around 0
    x = np.array([-2, -1, 0, 1, 2])
    result = Perceptron.tanh(x)
    assert np.allclose(result, -Perceptron.tanh(-x))  # Check if tanh(x) = -tanh(-x)


# ReLU Tests
def test_relu_positive_values():
    # Test positive input values
    x = np.array([1, 2.5, 10, 0.01])
    expected_output = np.array([1, 2.5, 10, 0.01])
    result = Perceptron.relu(x)
    assert np.array_equal(result, expected_output)


def test_relu_negative_values():
    # Test negative input values
    x = np.array([-1, -3.2, -100, -0.005])
    expected_output = np.array([0, 0, 0, 0])
    result = Perceptron.relu(x)
    assert np.array_equal(result, expected_output)


def test_relu_zero():
    # Test zero input
    x = np.array([0])
    expected_output = np.array([0])  # ReLU should output 0 for zero
    result = Perceptron.relu(x)
    assert np.array_equal(result, expected_output)

