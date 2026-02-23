import numpy as np

# +1 = "on" (white pixel), -1 = "off" (black pixel)

GRID_SIZE = 5  # each pattern is GRID_SIZE x GRID_SIZE

def make_pattern(grid):
    # grid is a 2D list of 0s and 1s (easy to read/write by hand)
    # Convert it to a 1D numpy array of +1 and -1
    result = np.array(grid).flatten()
    result = result * 2 - 1
    return result


def corrupt_pattern(pattern, noise_level=0.2):
    # pattern is a 1D array of +1/-1 with shape (N,)
    # noise_level is the fraction of neurons to flip (e.g. 0.2 = flip 20%)
    corrupted = pattern.copy()
    n_flips = int(noise_level * len(corrupted))
    indices = np.random.choice(len(corrupted), n_flips, replace=False)
    corrupted[indices] *= -1
    return corrupted




# Define some simple 5x5 patterns here using 0s and 1s
# Try to make them look like letters or shapes
# These will be the "memories" stored in the network

pattern_X = make_pattern([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
])

pattern_O = make_pattern([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

# Add more patterns here if you want — just keep the total count
# well below N/0.14 (where N=25) to avoid overloading the network.
# For a 25-neuron network that's roughly 3-4 patterns max.

ALL_PATTERNS = [pattern_X, pattern_O]
