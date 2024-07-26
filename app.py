from perceptron import Perceptron
import random


def perceptron_quality(pctron: Perceptron, test_count: int, logic: str) -> float:
    """
    Test the quality of the perceptron by checking its accuracy.
    """
    num_correct = 0
    all_possible_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _ in range(test_count):
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
    return (num_correct / test_count) * 100


def create_legend(*args: any) -> str:
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


# Change these parameters to affect the process.
epochs = 200
logic_to_apply = 'OR'
num_tests = 500
learning_rate = 0.01
activator = 'STEP'
use_threshold = False

legend = create_legend(epochs, logic_to_apply, learning_rate, num_tests, activator, use_threshold)

perceptron = Perceptron(logic_to_apply=logic_to_apply, learning_rate=learning_rate,
                        activator=activator, use_threshold=use_threshold)
perceptron.train(epochs, legend)

acc = perceptron_quality(perceptron, num_tests, logic_to_apply)
print(f"Accuracy: {acc:.2f}%")
