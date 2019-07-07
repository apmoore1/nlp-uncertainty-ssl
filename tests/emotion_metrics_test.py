import pytest
import numpy as np
from typing import List

from nlp_uncertainty_ssl.emotion_metrics import jaccard_index, f1_metric

@pytest.mark.parametrize("incl_neutral", (True, False))
def test_jaccard_index(incl_neutral: bool):
    # Simple case where there is not neutral
    predictions = np.array([[0,1,1,0,0], [0,0,0,1,0]])
    gold = np.array([[0,1,0,1,1], [0,0,0,1,0]])
    answer = (1/4 + 1) / 2
    assert answer == jaccard_index(predictions, gold, incl_neutral)
    # Case where the neutral class is wrong for predictions
    predictions = np.array([[0,1,1,0,0], [0,0,0,1,0], [0,1,1,0,0]])
    gold = np.array([[0,1,0,1,1], [0,0,0,1,0], [0,0,0,0,0]])
    if incl_neutral:
        answer = (1/4 + 1 + 0/3) / 3
        assert answer == jaccard_index(predictions, gold, incl_neutral)
    else:
        answer = (1/4 + 1 + 0/2) / 3
        assert answer == jaccard_index(predictions, gold, incl_neutral)
    # Case where the neutral case is True for predictions and gold
    predictions = np.array([[0,1,1,0,0], [0,0,0,1,0], [0,0,0,0,0]])
    gold = np.array([[0,1,0,1,1], [0,0,0,1,0], [0,0,0,0,0]])
    if incl_neutral:
        answer = (1/4 + 1 + 1) / 3
        assert answer == jaccard_index(predictions, gold, incl_neutral)
    else:
        answer = (1/4 + 1 + 0) / 3
        assert answer == jaccard_index(predictions, gold, incl_neutral)
    
    # Check that it raises assertions based on the size of the arrays
    predictions = np.array([0,1,1,0,0])
    gold = np.array([[0,1,0,1,1], [0,0,0,1,0]])
    with pytest.raises(AssertionError):
        jaccard_index(predictions, gold, incl_neutral)
    predictions = np.array([[0,1,0,1,1], [0,0,0,1,0]])
    gold = np.array([0,1,1,0,0]) 
    with pytest.raises(AssertionError):
        jaccard_index(predictions, gold, incl_neutral)


from typing import List
def f1_list(precisions: List[float], recalls: List[float]) -> List[float]:
    f1_values = []
    for precision, recall in zip(precisions, recalls):
        if precision == 0 and recall == 0:
            f1_values.append(0)
        else:
            f1 = (2 * precision * recall) / (precision + recall) 
            f1_values.append(f1)
    return f1_values

@pytest.mark.parametrize("incl_neutral", (True, False))
@pytest.mark.parametrize("macro", (True, False))
def test_f1_metric(incl_neutral: bool, macro: bool):
    def f1_list(precisions: List[float], recalls: List[float]) -> List[float]:
        f1_values = []
        for precision, recall in zip(precisions, recalls):
            if precision == 0 and recall == 0:
                f1_values.append(0.0)
            else:
                f1 = (2 * precision * recall) / (precision + recall) 
                f1_values.append(f1)
        return f1_values
    # Simple case where there is not neutral
    predictions = np.array([[1,0,0], [0,1,1]])
    gold = np.array([[1,1,0], [1,0,1]])

    precision_per_class = [1.0, 0.0, 1.0]
    recall_per_class = [0.5, 0.0, 1.0]
    num_classes = 3

    if incl_neutral:
        precision_per_class = [1.0, 0.0, 1.0, 0.0]
        recall_per_class = [0.5, 0.0, 1.0, 0.0]
        num_classes = 4
    
    
    if macro:
        f1_per_class = f1_list(precision_per_class, recall_per_class)
        gold_macro_f1 = sum(f1_per_class) / num_classes
        assert gold_macro_f1 == f1_metric(predictions, gold, macro, incl_neutral) 
    else:
        overall_precision = sum(precision_per_class) / 3
        overall_recall = sum(recall_per_class) / 3
        gold_micro = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
        assert gold_micro == f1_metric(predictions, gold, macro, incl_neutral)

    # Check that it raises assertions based on the size of the arrays
    predictions = np.array([0,1,1,0,0])
    gold = np.array([[0,1,0,1,1], [0,0,0,1,0]])
    with pytest.raises(AssertionError):
        f1_metric(predictions, gold, macro, incl_neutral)
    predictions = np.array([[0,1,0,1,1], [0,0,0,1,0]])
    gold = np.array([0,1,1,0,0]) 
    with pytest.raises(AssertionError):
        f1_metric(predictions, gold, macro, incl_neutral)