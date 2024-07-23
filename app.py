from perceptron import Perceptron
import numpy as np
import toml


"""
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
"""

def perceptron_quality(num_tests):
    num_correct = 0
    for _ in range(num_tests):
        input_data = np.random.randint(0, 2, size=2)

        # No conversion needed, input_data is already a NumPy array
        predicted_output = perceptron.predict(input_data)

        # Find the index using NumPy's array comparison
        index = np.where(np.all(perceptron.inputs == input_data, axis=1))[0]

        # Handle the case where input_data is not found in perceptron.inputs
        if len(index) == 0:
            continue

        expected_output = perceptron.get_expected_output()[index[0]]

        if predicted_output == expected_output:
            num_correct += 1

    return (num_correct / num_tests) * 100



cfg = toml.load('.streamlit/config.toml')
perceptron = Perceptron(logic_to_apply='AND', learning_rate=0.01)
perceptron.train(1000)

acc = perceptron_quality(500)
print(f"Accuracy: {acc:.2f}%")