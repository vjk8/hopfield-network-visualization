import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        # Store the number of neurons
        # Initialize a weight matrix of shape (n_neurons, n_neurons) filled with zeros
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        # patterns is a list of 1D numpy arrays, each with values +1 or -1
        # Use Hebbian learning to store each pattern into the weight matrix:
        #   For each pattern p: W += (1/N) * outer product of p with itself
        # After the loop, zero out the diagonal (no self-connections):
        N = self.weights.shape[0]
        for p in patterns:
            self.weights += (1/N) * np.outer(p, p)

        np.fill_diagonal(self.weights, 0)

    def update(self, state):
        # Compute the new state for all neurons at once (synchronous update)
        # new_state = sign( W dot state )
        # Return the new state
        new_state = np.sign(np.dot(self.weights, state))
        return new_state

    def energy(self, state):
        # Compute the network energy: E = -0.5 * state^T * W * state
        # Lower energy = more stable state (stored patterns are energy minima)
        # Return E
        E = state @ self.weights @ state
        E = -0.5 * E
        return E

    def recall(self, initial_state, steps=20):
        # Run the update rule for `steps` iterations starting from initial_state
        # Store and return the history of states (list of states at each step)
        # so we can animate the recall process later
        state = initial_state.copy()
        history = [state.copy()]
        for i in range(steps):
            state = self.update(state)
            history.append(state.copy())

        return history