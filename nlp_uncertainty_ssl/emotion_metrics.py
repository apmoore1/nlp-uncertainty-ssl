from typing import List

import numpy as np
from sklearn.metrics import jaccard_score, f1_score

def jaccard_index(predictions: np.ndarray, gold: np.ndarray,
                  incl_neutral: bool = False) -> float:
    '''
    :param predictions: Matrix of predictions of shape (n_samples, n_labels)
    :param gold: Matrix of gold labels of shape (n_samples, n_labels)
    :param incl_neutral: Whether to include the neutral label in the score 
                         where the neutral label is denoted by all the lables 
                         being 0
    :returns: The average jaccard_index score.
    :raises AssertionError: If the ``predictions`` or ``gold`` arguments are 
                            not a matrix e.g. do not have 2 dimensions. 
    '''
    assert len(predictions.shape) == 2
    assert len(gold.shape) == 2
    batch_size, num_labels = predictions.shape
    if incl_neutral:
        temp_predictions = np.zeros((batch_size, num_labels + 1))
        temp_predictions[:, :-1] = predictions
        is_neutral = (predictions.sum(axis=1) == 0) + 0.0
        temp_predictions[:,-1] = is_neutral
        predictions = temp_predictions

        temp_gold = np.zeros((batch_size, num_labels + 1))
        temp_gold[:, :-1] = gold
        is_neutral = (gold.sum(axis=1) == 0) + 0.0
        temp_gold[:,-1] = is_neutral
        gold = temp_gold
    return jaccard_score(gold, predictions, average='samples')

def f1_metric(predictions: np.ndarray, gold: np.ndarray,
              macro: bool = True, incl_neutral: bool = False) -> float:
    '''
    :param predictions: Matrix of predictions of shape (n_samples, n_labels)
    :param gold: Matrix of gold labels of shape (n_samples, n_labels)
    :param macro: If True then returns the macro F1 score else returns micro 
                  F1 score.
    :param incl_neutral: Whether to include the neutral label in the score 
                         where the neutral label is denoted by all the lables 
                         being 0
    :returns: The Macro/Micro F1 score.
    :raises AssertionError: If the ``predictions`` or ``gold`` arguments are 
                            not a matrix e.g. do not have 2 dimensions. 
    '''
    assert len(predictions.shape) == 2
    assert len(gold.shape) == 2
    batch_size, num_labels = predictions.shape
    if incl_neutral:
        temp_predictions = np.zeros((batch_size, num_labels + 1))
        temp_predictions[:, :-1] = predictions
        is_neutral = (predictions.sum(axis=1) == 0) + 0.0
        temp_predictions[:,-1] = is_neutral
        predictions = temp_predictions

        temp_gold = np.zeros((batch_size, num_labels + 1))
        temp_gold[:, :-1] = gold
        is_neutral = (gold.sum(axis=1) == 0) + 0.0
        temp_gold[:,-1] = is_neutral
        gold = temp_gold
    # If not macro performs micro
    if macro:
        return f1_score(gold, predictions, average='macro')
    else:
        return f1_score(gold, predictions, average='micro')