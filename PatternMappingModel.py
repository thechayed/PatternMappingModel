import random
import numpy as np
import math

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
    if is_list(integer):
        return integer
    bits = []

    integer = int(integer)

    while integer:
        bit = int(integer) & 1
        bits.append(bit)
        
        integer >>= 1

    if not bits:
        bits.append(0)

    # bits.reverse()
    return bits

def bits_to_integer(bits):
    if is_integer(bits):
        return bits
    result = 0
    reversed = bits[::-1]
    for bit in reversed:
        result = (result << 1) | int(round(bit))
    return result

def subtract_bits_by_floats(integer, floats_array):
    integer_bits = integer_to_bits(integer)
    # floats_to_int = bits_to_integer(np.int16(floats_array))

    first = np.pad(np.array(integer_bits, dtype=np.float16), (0, max(len(floats_array) - len(integer_bits), 0)), mode='constant')
    second = np.pad(np.array(floats_array, dtype=np.float16), (0, max(len(integer_bits) - len(floats_array), 0)), mode='constant')

    return (np.abs(np.mean(np.subtract(first, second)))+0.01) * (abs(integer - bits_to_integer(np.int16(floats_array))) ** 5) * ((np.abs(np.mean(first) - np.mean(second))+0.01)**2)

def bubble_by_score(array_to_sort, scores_and_indices, descending):
    isSorted = False
    
    while not isSorted:
        isSorted = True
        if descending:
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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def random_partition(total_value, num_elements):
    partition = []
    remaining_value = total_value

    for _ in range(num_elements - 1):
        # Generate a random value for the current element
        element =  random.uniform(-random.uniform(0, remaining_value),random.uniform(0, remaining_value))
        partition.append(element)
        remaining_value -= abs(-element)

    # The last element is whatever remains to reach the total value
    partition.append(remaining_value)

    return partition

# Example usage

class PatternMappingModel:
    def __init__(self, input_size, output_size, scoring_falloff, precision, mutation_rate, appraisal_strength, punishment_strength):
        self.input_size = input_size
        self.output_size = output_size
        self.scoring_falloff = scoring_falloff
        self.precision = precision
        self.mutation_rate = mutation_rate
        self.appraisal_strength = appraisal_strength
        self.punishment_strength = punishment_strength

        # Generate all possible input combinations
        self.input_combinations = 2 ** input_size 
        self.output_combinations = 2 ** output_size

        # Relates Inputs and Outputs by Weights
        self.inference_map = np.empty((self.input_combinations), dtype=object)

        for input_combination in range(self.input_combinations):
            self.inference_map[input_combination] = {}

            for output_combination in range(self.output_combinations):
                self.inference_map[input_combination][output_combination] = random.uniform(0.0,1.0)
        
        # Stores factors for each input, that are multiplied into the value gathered from the Inference Map on Inference
        self.input_perturbation_map = np.empty((self.input_combinations), dtype=object)
        for input_combination in range(self.input_combinations):
            self.input_perturbation_map[input_combination] = np.zeros(self.output_size)

    def inference(self, input):
        input_index = None

        if is_integer(input):
            input_index = input
        if is_list(input):
            input_index = bits_to_integer(input)

        merged_output = np.zeros(self.output_size, dtype=np.float16)

        # Merge the Final Output Weights
        outputs_weights_normalized = self.normalize_input(input_index)
        for output in outputs_weights_normalized.keys():
            output_bits = integer_to_bits(output)
            output_bits = np.pad(integer_to_bits(output), (0, len(merged_output) - len(output_bits)), mode='constant')
            for bit in range(len(output_bits)):
                merged_output[bit] = lerp(merged_output[bit], output_bits[bit], outputs_weights_normalized[output])

        for j in range(len(merged_output)):
            merged_output[j] = sigmoid(merged_output[j] + self.input_perturbation_map[input_index][j]) 

        return merged_output

    def praise(self, input, output, power):
        input_index = bits_to_integer(input)
        output_index = bits_to_integer(output)

        sorted_inputs = self.get_closest_inputs(input_index)

        for i in range(len(sorted_inputs)-1):
            self.inference_map[sorted_inputs[i]][output_index] += (self.appraisal_strength * power * scoring_falloff ** i) + random.uniform(-self.mutation_rate, self.mutation_rate)

    # Creates Perturbations to the Inferred Output Calculation, based on the amount of error that is present in the output
    def punish(self, input, power):
        input_index = None
        output_bits = None
        
        input_index = bits_to_integer(input)

        error = power * self.punishment_strength
        partitioned_error = random_partition(error, self.input_size)

        for i in partitioned_error:
            i = sigmoid(i)

        self.input_perturbation_map[input_index] += partitioned_error


    # Normalizes the Weighted Relationships of the Outputs associated with the given Input
    def normalize_input(self, input):
        max_weight = 0
        normalized = {}

        for output_index in range(self.output_combinations):
            if  self.inference_map[input][output_index] > max_weight:
                max_weight = self.inference_map[input][output_index]

        for output_index in range(self.output_combinations):
            normalized[output_index] = (self.inference_map[input][output_index]/max_weight) ** 10

        return normalized
    
    def get_closest_inputs(self, input):
        input_index = bits_to_integer(input)
        
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

        return sorted_inputs

    def train(self, dataset):
        for input, output in dataset.items():
            self.praise(input, output,1)

# Example usage
input_size = 5
output_size = 5
scoring_falloff = 0.15
mutation_rate = 0.2
precision = 3
appraisal_strength = 1
punishment_strength = 1

model = PatternMappingModel(input_size, output_size, scoring_falloff, precision, mutation_rate, appraisal_strength, punishment_strength)

# Define your model and dataset here

# Define your dataset, for example:
dataset = {
    1: 1,
    2: 4,
    3: 9,
    4: 16
    # Add more data as needed
}

# Train the model
for i in range(10):
    model.train(dataset)

# Evaluate the model on the test dataset
correct_predictions = 0
total_predictions = len(dataset)

print(bits_to_integer(model.inference(2)))

#Train until we get the value 25 for the input 5

power_reached = False
while not power_reached:
    predicted = bits_to_integer(model.inference(5))
    model.punish(5, abs(25 - predicted) * 3)
    print(bits_to_integer(model.inference(5)))

    power_reached = predicted == 25

print(bits_to_integer(model.inference(2)))

