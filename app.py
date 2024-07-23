from perceptron import Perceptron
import numpy as np
import toml
import random


def perceptron_quality(pctron, num_tests, logic):
    num_correct = 0
    all_possible_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _ in range(num_tests):
        input_data = random.choice(all_possible_inputs)
        predicted_output = pctron.predict(input_data)

        # Calculate expected output based on the logic being trained
        if logic == 'AND':
            expected_output = 1 if all(input_data) else 0
        elif logic == 'OR':
            expected_output = 1 if any(input_data) else 0
        else:
            raise ValueError("Invalid logic_to_apply")

        if predicted_output == expected_output:
            num_correct += 1
    return (num_correct / num_tests) * 100


epochs = 200
# Ensure the logic is consistent during training and evaluation
logic_to_apply = 'AND'  # Or 'AND' if you want to train for AND
perceptron = Perceptron(logic_to_apply=logic_to_apply, learning_rate=0.01, activator='STEP', use_threshold=False)
perceptron.train(epochs)

acc = perceptron_quality(perceptron, 500, logic_to_apply)
print(f"Accuracy: {acc:.2f}%")



