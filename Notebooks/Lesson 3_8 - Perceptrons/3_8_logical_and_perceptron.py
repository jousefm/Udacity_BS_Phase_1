import pandas as pd

weight1 = 1.5
weight2 = 1.5
bias = -2


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)] # coordinates
correct_outputs = [False, False, False, True]  # labels
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias # Calculate the linear equation
    output = int(linear_combination >= 0)  # Evaluate the result ( step function )
    is_correct_string = 'Yes' if output == correct_output else 'No'
    # Append the inputs, the evaluation, Activation Output and Is correct
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
# Create a dataframe to print it nicely
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))