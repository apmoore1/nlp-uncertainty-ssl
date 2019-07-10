from typing import List, Dict, Any, Tuple
import argparse
import json
from pathlib import Path

import numpy as np

from nlp_uncertainty_ssl import model_util
from nlp_uncertainty_ssl import emotion_metrics

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def read_dataset(dataset_fp: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    samples = []
    labels = []
    with dataset_fp.open('r') as dataset_file:
        for line in dataset_file:
            sample = json.loads(line)
            labels.extend(sample['labels'])
            samples.append(sample)
    labels = set(labels)
    labels = sorted(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    new_samples = []
    for sample in samples:
        label_array = [0] * len(label_to_index)
        for label in sample['labels']:
            label_array[label_to_index[label]] = 1
        sample['label_array'] = label_array
        new_samples.append(sample)
    return new_samples, label_to_index

def majority_class(train_dataset: List[Dict[str, Any]],
                   test_dataset: List[Dict[str, Any]]) -> np.ndarray:
    label_array = model_util.to_label_arrays(train_dataset, 'label_array')
    label_counts = label_array.sum(axis=0)
    majority_index = np.argmax(label_counts)
    
    test_label_array = model_util.to_label_arrays(test_dataset, 'label_array')
    majority_array = np.zeros_like(test_label_array)
    majority_array[:, majority_index] = 1
    return majority_array

def random_class(test_dataset: List[Dict[str, Any]]) -> np.ndarray:
    test_label_array = model_util.to_label_arrays(test_dataset, 'label_array')
    array_size = test_label_array.shape
    random_array = np.random.randint(2, size=array_size)
    return random_array

def arrays_scores(prediction_array: np.ndarray, gold_array: np.ndarray,
                  name: str) -> Dict[str, float]:
    jaccard_index = emotion_metrics.jaccard_index(prediction_array, gold_array, incl_neutral=True)
    macro_f1 = emotion_metrics.f1_metric(prediction_array, gold_array, macro=True, incl_neutral=True)
    micro_f1 = emotion_metrics.f1_metric(prediction_array, gold_array, macro=False, incl_neutral=True)
    score_dict = {f'{name} jaccard index': jaccard_index,
                  f'{name} macro f1': macro_f1, f'{name} micro f1': micro_f1}
    return score_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset_fp', type=parse_path, 
                        help='Training Dataset')
    parser.add_argument('test_dataset_fp', type=parse_path, 
                        help='Test Dataset')
    args = parser.parse_args()

    train_dataset, train_label_index = read_dataset(args.train_dataset_fp)
    test_dataset, test_label_index = read_dataset(args.test_dataset_fp)
    print(f'Length of train and test: {len(train_dataset)} {len(test_dataset)}')
    print(f'Train label to index: {train_label_index}')
    print(f'Test label to index: {test_label_index}')

    test_gold_array = model_util.to_label_arrays(test_dataset, 'label_array')
    majority_array = majority_class(train_dataset, test_dataset)
    print(arrays_scores(majority_array, test_gold_array, 'Majority'))
    random_array = random_class(test_dataset)
    print(arrays_scores(random_array, test_gold_array, 'Random'))