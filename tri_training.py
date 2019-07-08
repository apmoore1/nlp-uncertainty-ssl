import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional
import random
import statistics

import numpy as np

from nlp_uncertainty_ssl import model_util
from nlp_uncertainty_ssl import emotion_metrics

def index_array_to_labels(index_array: List[int], index_to_label: Dict[int, str]) -> List[str]:
    labels = []
    for label_index, is_label in enumerate(index_array):
        if is_label:
            labels.append(index_to_label[label_index])
    return labels

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def read_dataset(dataset_fp: Path) -> List[Dict[str, Any]]:
    samples = []
    with dataset_fp.open('r') as dataset_file:
        for line in dataset_file:
            samples.append(json.loads(line))
    return samples

def data_to_file(data: List[Dict[str, Any]], fp: Path) -> None:
    with fp.open('w+') as _file:
        for index, sample in enumerate(data):
            sample = json.dumps(sample)
            if index != 0:
                _file.write(f'\n{sample}')
            else:
                _file.write(f'{sample}')


def read_remove_write(unlabelled_data_fp: Path, temp_fp: Path):
    '''
    Changes the labels from the unlablled data to no label (neutral) of which 
    the data is written to the temp_fp and the original unlabelled data 
    labels are not changed in the file.
    '''
    unlabelled_data = []
    with unlabelled_data_fp.open('r') as unlabelled_file:
        for line in unlabelled_file:
            sample = json.loads(line)
            sample['labels'] = []
            unlabelled_data.append(sample)
    with temp_fp.open('w+') as temp_file:
        for index, sample in enumerate(unlabelled_data):
            line = json.dumps(sample)
            if index != 0:
                temp_file.write(f'\n{line}')
            else:
                temp_file.write(f'{line}')

def ensure_label_vocab_same(label_vocabs: List[Dict[str, int]]):
    assert len(label_vocabs) == 3
    label_vocab_1, label_vocab_2, label_vocab_3 = label_vocabs
    for key, value in label_vocab_1.items():
        assert value == label_vocab_2[key]
        assert value == label_vocab_3[key]
    assert len(label_vocab_1) == len(label_vocab_2)
    assert len(label_vocab_1) == len(label_vocab_3)

def majority_score_preds(preds: List[List[Dict[str, Any]]]) -> np.ndarray:
    assert len(preds) == 3
    preds_1, preds_2, preds_3 = preds
    assert len(preds_1) == len(preds_2)
    assert len(preds_2) == len(preds_3)
    majority_preds = []
    for i in range(len(preds_1)):
        pred_1 = preds_1[i]['prediction']
        pred_2 = preds_2[i]['prediction']
        pred_3 = preds_3[i]['prediction']
        if pred_1 == pred_2 == pred_3:
            majority_preds.append(pred_1)
            continue
        
        if pred_1 == pred_2:
            majority_preds.append(pred_1)
        elif pred_2 == pred_3:
            majority_preds.append(pred_2)
        elif pred_1 == pred_3:
            majority_preds.append(pred_1)
        else:
            majority_preds.append(random.choices([pred_1, pred_2, pred_3])[0])
    assert len(majority_preds) == len(preds_1)
    return np.array(majority_preds)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", type=parse_path,
                        help="File path to the training dataset")
    parser.add_argument("dev_fp", type=parse_path,
                        help="File path to the development dataset")
    parser.add_argument("test_fp", type=parse_path,
                        help='File path to the test dataset')
    parser.add_argument("unlabelled_fp", type=parse_path,
                        help='Unlabelled dataset')
    parser.add_argument("save_dir", type=parse_path, help='Save directory')
    parser.add_argument("model_fps", type=parse_path, nargs='+',
                        help='File path to the model config file(s)')
    args = parser.parse_args()
    train_fp = args.train_fp
    dev_fp = args.dev_fp
    test_fp = args.test_fp
    model_fps = args.model_fps
    unlabelled_fp = args.unlabelled_fp
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_dev_results_fp = Path(args.save_dir, 'dev.json').resolve()
    save_test_results_fp = Path(args.save_dir, 'test.json').resolve()

    best_dev_scores = []
    best_test_scores = []
    for i in range(5):
        print(best_dev_scores)
        print(best_test_scores)
        # Make sure the unlabelled data has no labels
        with tempfile.TemporaryDirectory() as temp_dir:
            # Remove labels from unlabelled data
            temp_unlabelled_data_path = Path(temp_dir, 'unlabelled_data.json').resolve()
            read_remove_write(unlabelled_fp, temp_unlabelled_data_path)
            # Train and unlabelled dataset per classifier
            train_datas = [read_dataset(train_fp), read_dataset(train_fp), read_dataset(train_fp)]
            #unlabelled_datas = [read_dataset(temp_unlabelled_data_path), 
            #                    read_dataset(temp_unlabelled_data_path),
            #                    read_dataset(temp_unlabelled_data_path)]
            # These are never changed
            dev_data = read_dataset(dev_fp)
            test_data = read_dataset(test_fp)
            

            majority_better = True
            best_dev_score = 0
            best_test_score = 0
            while majority_better:
                model_names = []
                model_dev_preds = []
                model_test_preds = []
                model_unlabelled_predictions = []
                model_index_label_vocabs = []
                model_vocab = None
                # Train each model
                for model_index, model_fp in enumerate(model_fps):
                    model_name = model_fp.stem
                    model_names.append(model_name)
                    # Get model specific train fp
                    model_train_fp = Path(temp_dir, 'temp_train_file.json').resolve()
                    data_to_file(train_datas[model_index], model_train_fp)
                    # Train model
                    model, model_params = model_util.train_model(model_train_fp, dev_fp, model_fp, 
                                                                vocab_data_fps=[test_fp, temp_unlabelled_data_path])
                    # just in case
                    model_index_label_vocabs.append(model.vocab.get_index_to_token_vocabulary(namespace='labels'))
                    # Model dev predictions
                    model_dev_preds.append(model_util.predict(model, model_params, dev_fp))
                    # Model dev predictions
                    model_test_preds.append(model_util.predict(model, model_params, test_fp))
                    # Model unlabelled predictions
                    #model_unlabelled_fp = Path(temp_dir, 'temp_unlablled_file.json').resolve()
                    #data_to_file(unlabelled_datas[model_index], model_unlabelled_fp)
                    #unlabelled_datas[model_index] = model_util.predict(model, model_params, model_unlabelled_fp)
                    model_unlabelled_predictions.append(model_util.predict(model, model_params, temp_unlabelled_data_path))
                    model_vocab = model.vocab
                # Ensure the label vocabs are the same
                #ensure_label_vocab_same(model_label_vocabs)
                # Dev score
                dev_preds_array = majority_score_preds(model_dev_preds)
                dev_gold_array = model_util.to_label_arrays(model_util.read_dataset(dev_fp, True, model_vocab), 'label_array')
                dev_score = emotion_metrics.jaccard_index(dev_preds_array, dev_gold_array, True)
                # Test score
                test_preds_array = majority_score_preds(model_test_preds)
                test_gold_array = model_util.to_label_arrays(model_util.read_dataset(test_fp, True, model_vocab), 'label_array')
                test_score = emotion_metrics.jaccard_index(test_preds_array, test_gold_array, True)
                
                print(dev_score)
                print(test_score)

                if best_dev_score < dev_score:
                    best_dev_score = dev_score
                    best_test_score = test_score
                else:
                    best_dev_scores.append(best_dev_score)
                    best_test_scores.append(best_test_score)
                    majority_better = False
                    break
                # Perform tri-training predictions
                index_to_label = model_vocab.get_index_to_token_vocabulary('labels')
                for model_index_add, model_index_1, model_index_2 in [[0,1,2], [1,2,0], [2,1,0]]:
                    model_unlabelled_predictions_add = model_unlabelled_predictions[model_index_add]
                    model_unlabelled_predictions_1 = model_unlabelled_predictions[model_index_1]
                    model_unlabelled_predictions_2 = model_unlabelled_predictions[model_index_2]

                    # index to labels
                    index_label_add = model_index_label_vocabs[model_index_add]
                    index_label_1 = model_index_label_vocabs[model_index_1]
                    index_label_2 = model_index_label_vocabs[model_index_2]

                    model_train_data_add = train_datas[model_index_add]
                    model_train_data_add_ids = [sample['ID'] for sample in model_train_data_add]
                    for i in range(len(model_unlabelled_predictions_add)):
                        sample_add = model_unlabelled_predictions_add[i]
                        if sample_add['ID'] in model_train_data_add_ids:
                            continue
                        sample_1_pred = model_unlabelled_predictions_1[i]['prediction']
                        sample_1_pred = sorted(index_array_to_labels(sample_1_pred, index_label_1))
                        sample_2_pred = model_unlabelled_predictions_2[i]['prediction']
                        sample_2_pred = sorted(index_array_to_labels(sample_2_pred, index_label_2))
                        if sample_1_pred == sample_2_pred:
                            sample_add_pred = sorted(index_array_to_labels(sample_add['prediction'], index_label_add))
                            if sample_1_pred != sample_add_pred:
                                sample_add['labels'] = sample_add_pred
                                model_train_data_add.append(sample_add)
    
    dev_scores = statistics.mean(best_dev_scores)
    dev_scores = {'jaccard_index': dev_scores}
    test_scores = statistics.mean(best_test_scores)
    test_scores = {'jaccard_index': test_scores}
    print(dev_scores)
    print(test_scores)
    with save_dev_results_fp.open('w+') as dev_results_file:
        json.dump(dev_scores, dev_results_file)
    with save_test_results_fp.open('w+') as test_results_file:
        json.dump(test_scores, test_results_file)