# Imports required to support the kNN modules.
import statistics

# Modules required for kNN algorithm.

# Points as [x, y, z, ...]
def euclidean_distance(point_1, point_2):
    if len(point_1) != len(point_2):
        print("Arrays are not compatible shapes.")
        return None

    s = 0.0
    for i in range(len(point_1)):
        s += ((point_1[i] - point_2[i]) ** 2)
    return s ** 0.5


def KNN(query, known_inputs, known_outputs, k_number, show_neighbours = False):
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

    if show_neighbours:
        print(shortest)
    
    # 4. Use mode to predict the unknown output.
    output = statistics.mode(shortest)
    return output  

