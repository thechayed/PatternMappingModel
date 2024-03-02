import random
import numpy as np
import pickle

def is_integer(variable):
    return isinstance(variable, int)

# Check if a variable is a list
def is_list(variable):
    return isinstance(variable, list) | isinstance(variable, np.ndarray) | isinstance(variable, tuple)

def lerp(a, b, t):
    a = float(a)
    b = float(b)
    t = float(t)
    return a * (1 - t) + b * t

def integer_to_bits(integer):
    bits = []

    integer = int(integer)

    while integer:
        bit = int(integer) & 1
        bits.append(bit)
        
        integer >>= 1

    if not bits:
        bits.append(0)

    bits.reverse()
    return bits

def bits_to_integer(bits):
    result = 0
    reversed = bits[::-1]
    for bit in reversed:
        result = (result << 1) | int(round(bit))
    return result

def subtract_bits_by_floats(integer, floats_array):
    floats_to_int = bits_to_integer(np.int16(floats_array))
    return abs(floats_to_int - integer)

def bubble_by_score(array_to_sort, scores_and_indices, decending):
    isSorted = False
    
    while not isSorted:
        isSorted = True
        if decending:
            for i in range(len(array_to_sort) - 1):
                if scores_and_indices[array_to_sort[i]][0] > scores_and_indices[array_to_sort[i + 1]][0]:  # Change to '>' for descending order
                    temp = array_to_sort[i]
                    array_to_sort[i] = scores_and_indices[array_to_sort[i + 1]][1]
                    array_to_sort[i + 1] = scores_and_indices[temp][1]
                    isSorted = False
        else:
            for i in range(len(array_to_sort) - 1):
                if scores_and_indices[array_to_sort[i]][0] < scores_and_indices[array_to_sort[i + 1]][0]:  # Change to '>' for descending order
                    temp = array_to_sort[i]
                    array_to_sort[i] = scores_and_indices[array_to_sort[i + 1]][1]
                    array_to_sort[i + 1] = scores_and_indices[temp][1]
                    isSorted = False
    return array_to_sort

class PatternMappingModel:
    def __init__(self, input_size, output_size, scoring_falloff, precision, mutation_rate, appraisal_strength):
        self.input_size = input_size
        self.output_size = output_size
        self.scoring_falloff = scoring_falloff
        self.precision = precision
        self.mutation_rate = mutation_rate
        self.appraisal_strength = appraisal_strength

        # Generate all possible input combinations
        self.input_combinations = 2 ** input_size 
        self.output_combinations = 2 ** output_size

        self.inference_map = np.empty(2 ** self.input_size, dtype=object)
        # Inputs to Outputs
        for input_combination in range(self.input_combinations):
            self.inference_map[input_combination] = np.zeros(self.output_combinations, dtype=np.float16)

    def inference(self, input):
        input_index = None

        if is_integer(input):
            input_index = input
        if is_list(input):
            input_index = bits_to_integer(input)

        output = np.array(self.inference_map[input_index])
        for i in range(len(output)):
            output[i] += random.uniform(-self.mutation_rate, self.mutation_rate)
        
        return output


    def praise_model(self, input, output, power):
        input_index = None
        output_bits = None
        
        if is_integer(input):
            input_index = input
        if is_list(input):
            input_index = bits_to_integer(input)

        if is_integer(output):
            output_bits = integer_to_bits(output)
        if is_list(output):
            output_bits = output

        # Calculate similarity scores between input_combination and all other input combinations
        similarity_scores = np.zeros((2 ** self.input_size, 2), dtype=np.float16)
        sorted_inputs = np.zeros(2 ** self.input_size, dtype=np.float16)

        for row_index in range(self.inference_map.shape[0]):
            similarity_scores[row_index][0] = np.mean(subtract_bits_by_floats(row_index, integer_to_bits(input_index)))
            similarity_scores[row_index][1] = row_index

        # Get the indices that would sort similarity_scores
        sorted_inputs = np.array(similarity_scores[:,1], dtype=int)

        # Bubble sort the inputs based on their similarity scores
        bubble_by_score(sorted_inputs, similarity_scores, True)

        for k in range(min(len(self.inference_map[input_index]), len(output_bits))):
            self.inference_map[input_index][k] = lerp(self.inference_map[input_index][k], output_bits[k], self.appraisal_strength * power)

        for i in range(len(sorted_inputs)-1):
            for k in range(min(len(self.inference_map[sorted_inputs[i]]), len(output_bits))):
                # Increase the score for the given output
                self.inference_map[sorted_inputs[i]][k] = lerp(self.inference_map[sorted_inputs[i]][k], self.inference_map[input_index][k], self.appraisal_strength * power * scoring_falloff ** i)


    def train(self, dataset):
        for input, output in dataset.items():
            self.praise_model(input, output, 1)

# Example usage
input_size = 5
output_size = 5
scoring_falloff = 0.25
mutation_rate = 0.25
precision = 3

model = PatternMappingModel(input_size, output_size, scoring_falloff, precision, mutation_rate, 1)

# Define your model and dataset here

# Define your dataset, for example:
dataset = {
    (1, 0, 1, 0, 1): (0, 1, 0, 1, 0),
    (0, 1, 0, 1, 0): (1, 0, 1, 0, 1),
    (1, 1, 1, 0, 1): (0, 1, 0, 1, 0),
    (0, 1, 0, 1, 1): (1, 0, 1, 0, 1),
    (1, 1, 1, 1, 1): (0, 1, 0, 1, 0),
    (0, 1, 1, 1, 1): (1, 0, 1, 0, 1),
    # Add more data as needed
}

# Split the dataset into training and testing sets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(dataset) * split_ratio)

# Train the model
for i in range(1000):
    model.train(dataset)

# Evaluate the model on the test dataset
correct_predictions = 0
total_predictions = len(dataset)

for i in range(100):
    correct_predictions = 0
    accuracy = 1

    for input_data, expected_output in dataset.items():
        predicted_output = model.inference(input_data)
        expected_output = np.pad(expected_output, (0, len(predicted_output) - len(expected_output)), mode='constant')
        accuracy = lerp(accuracy, 1 - (np.abs(np.mean(predicted_output - expected_output))), 1/(2 ** output_size))

    # accuracy = (correct_predictions / total_predictions) * 100  # Convert to percentage
    print("Accuracy:", round(accuracy * 100), "%")