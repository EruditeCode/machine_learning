# A program script to explore a basic kNN implementation.

# Walkthrough Video Link: https://www.youtube.com/channel/UC7pDQuoNVBujMYpzPREh5QA

import statistics


# Points as [x, y, z, ...]
def euclidean_distance(point_1, point_2):
    if len(point_1) != len(point_2):
        print("Arrays are not compatible shapes.")
        return None

    s = 0.0
    for i in range(len(point_1)):
        s += ((point_1[i] - point_2[i]) ** 2)
    return s ** 0.5


def KNN(query, known_inputs, known_outputs, k_number):
    # 1. Measure distances.
    distances = [euclidean_distance(query, p) for p in known_inputs]
    
    # 2. Select k number of shortest distances.
    shortest = [0 for i in range(k_number)]
    placeholder = max(distances)
    for i in range(k_number):
        min_index = distances.index(min(distances))
        # 3. Find the output associated with the short distance.
        shortest[i] = known_outputs[min_index]
        distances[min_index] = placeholder
    print(shortest)
    
    # 4. Use mode to predict the unknown output.
    output = statistics.mode(shortest)
    return output  


# Inputs are [height, diameter]
known_inputs = [[35, 27], [33, 26], [36, 25], [34, 27],
                [33, 28], [29, 23],	[45, 30], [43, 30],
                [40, 28], [45, 32], [48, 33], [46, 29]]
# Outputs are 1 for Quail, 2 for Chicken.
known_outputs = [1, 1, 1, 1, 1, 1,
				 2, 2, 2, 2, 2, 2]

test_point = [39, 29]
test = KNN(test_point, known_inputs, known_outputs, 3)
print(test)
