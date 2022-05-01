"""
hw2_decoder.py

This module is for implementations of the Viterbi and Forward-Backward
algorithms, which are used in Questions 1. Forward-Backward is provided
for you; you will need to implement Viterbi for Question 1.
In Question 2 you will adapt these algorithms to PyTorch.
"""

import numpy as np
from torch import log_


def viterbi(initial_scores, transition_scores, final_scores, emission_scores):
    """Computes the viterbi trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    Your solution should return:
    - best_path: (length) array containing the most likely state sequence
    """

    length = emission_scores.shape[0]
    num_states = initial_scores.shape[0]

    # Forward variables.
    forward = np.full((length, num_states), -np.inf)
    forward[0] = emission_scores[0] + initial_scores
    f_sequence = np.full((length, num_states), -np.inf)

    # Forward pass.
    for i in range(1, length):
        for state in range(num_states):
            forward[i, state] = np.max(forward[i - 1] + transition_scores[i - 1, state])
            forward[i, state] += emission_scores[i, state]
            f_sequence[i, state] = np.argmax(forward[i - 1] + transition_scores[i - 1, state])

    print(np.exp(forward))
    print('\n')

    print(f'viterbi forward: {forward}')
    print(f'sequence: {f_sequence}\n')

    # Backward variables.
    best_path = np.full((length,), -np.inf)
    best_path[length-1] = np.argmax(final_scores + forward[length-1,:])

    # Backward pass.
    for i in range(length-2, -1, -1):
        best_path[i] = f_sequence[i+1,int(best_path[i+1])]
            
    return best_path
    


def _forward(initial_scores, transition_scores, final_scores, emission_scores):
    """Compute the forward trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    """
    length = emission_scores.shape[0]
    num_states = initial_scores.shape[0]

    # Forward variables.
    forward = np.full((length, num_states), -np.inf)
    forward[0] = emission_scores[0] + initial_scores

    # Forward loop.
    for i in range(1, length):
        for state in range(num_states):
            forward[i, state] = np.logaddexp.reduce(
                forward[i - 1] + transition_scores[i - 1, state])
            forward[i, state] += emission_scores[i, state]


    # Termination.
    log_likelihood = np.logaddexp.reduce(forward[length - 1] + final_scores)

    print(f'\nlog likelihood: {log_likelihood}')
    print(f'\nforward: {forward}')

    return log_likelihood, forward


def _backward(
        initial_scores, transition_scores, final_scores, emission_scores):
    """Compute the backward trellis for a given sequence.
    Receives:
    - Initial scores: (num_states) array
    - Transition scores: (length-1, num_states, num_states) array
    - Final scores: (num_states) array
    - Emission scores: (length, num_states) array.
    """
    length = emission_scores.shape[0]
    num_states = initial_scores.shape[0]

    # Backward variables.
    backward = np.full((length, num_states), -np.inf)

    # Initialization.
    backward[length-1, :] = final_scores

    # Backward loop.
    for i in range(length - 2, -1, -1):
        for state in range(num_states):
            backward[i, state] = np.logaddexp.reduce(
                backward[i + 1] +
                transition_scores[i, :, state] +
                emission_scores[i + 1])

    # Termination.
    log_likelihood = np.logaddexp.reduce(
        backward[0, :] + initial_scores + emission_scores[0, :])

    print(f'\nlog likelihood: {log_likelihood}')
    print(f'\nbackward: {backward}\n')

    return log_likelihood, backward


def mrd(initial_scores, transition_scores, final_scores, emission_scores):

    _, forward = _forward(
        initial_scores, transition_scores, final_scores, emission_scores)
    _, backward = _backward(
        initial_scores, transition_scores, final_scores, emission_scores)
    
    length = np.size(emission_scores, 0)
    best_path = np.full((length,), -np.inf)

    for i in range(0, length):
        best_path[i] = np.argmax(np.array([forward[i,0]+backward[i,0], forward[i,1]+backward[i,1], forward[i,2]+backward[i,2]]))
        print(np.array([forward[i,0]+backward[i,0], forward[i,1]+backward[i,1], forward[i,2]+backward[i,2]]))

    return best_path



def forward_backward(
        initial_scores, transition_scores, final_scores, emission_scores):
    log_likelihood, forward = _forward(
        initial_scores, transition_scores, final_scores, emission_scores)

    log_likelihood, backward = _backward(
        initial_scores, transition_scores, final_scores, emission_scores)

    emission_posteriors = np.exp(forward + backward - log_likelihood)
    transition_posteriors = np.zeros_like(transition_scores)
    length = np.size(emission_scores, 0)  # Length of the sequence.
    # num_states = np.size(initial_scores)  # Number of states.
    for i in range(1, length):
        # bp: afaik, the multiplication by np.ones is unnecessary
        # transition_posteriors[i - 1] =
        # np.exp(forward[i - 1: i].transpose() +
        # transition_scores[i - 1] +
        # emission_scores[i: i + 1] +
        # backward[i: i + 1] -
        # log_likelihood)
        
        fw_t = forward[i - 1: i].T
        bw = backward[i]
        tr = transition_scores[i - 1]
        em = emission_scores[i]
        transition_posteriors[i - 1] = fw_t + tr + em + bw
    transition_posteriors = np.exp(transition_posteriors - log_likelihood)
    # the transition_posteriors aren't even used in Q2...
    print(transition_posteriors)

    return emission_posteriors, transition_posteriors, log_likelihood
