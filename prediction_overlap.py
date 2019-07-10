import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional

from nlp_uncertainty_ssl import model_util

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

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

def prediction_overlap(predicted_data_1: List[Dict[str, Any]],
                       predicted_data_2: List[Dict[str, Any]],
                       predicted_data_3: Optional[List[Dict[str, Any]]] = None
                       ) -> int:
    assert len(predicted_data_1) == len(predicted_data_2)
    if predicted_data_3 is not None:
        assert len(predicted_data_1) == len(predicted_data_3)
    same_prediction_count = 0
    for i in range(len(predicted_data_1)):
        prediction_1 = predicted_data_1[i]['prediction']
        prediction_2 = predicted_data_2[i]['prediction']
        if predicted_data_3 is not None:
            prediction_3 = predicted_data_3[i]['prediction']
            if prediction_1 == prediction_2 and prediction_1 == prediction_3:
                same_prediction_count += 1
        else:
            if prediction_1 == prediction_2:
                same_prediction_count += 1
    return same_prediction_count

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
    parser.add_argument("model_fps", type=parse_path, nargs='+',
                        help='File path to the model config file(s)')
    args = parser.parse_args()
    train_fp = args.train_fp
    dev_fp = args.dev_fp
    test_fp = args.test_fp
    model_fps = args.model_fps
    unlabelled_fp = args.unlabelled_fp
    # Make sure the unlabelled data has no labels
    with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
        temp_file_path = Path(temp_file.name)
        read_remove_write(unlabelled_fp, temp_file_path)

        model_names = []
        model_dev_scores = defaultdict(lambda: {})
        model_unlabelled_predictions = []
        # Train each model
        for model_fp in model_fps:
            model_name = model_fp.stem
            model_names.append(model_name)
            model, model_params = model_util.train_model(train_fp, dev_fp, model_fp, 
                                                            vocab_data_fps=[test_fp, temp_file_path])
            print(model.vocab.get_token_to_index_vocabulary(namespace='labels'))
            for score_name, value in model_util.dataset_scores(dev_fp, model, model_params).items():
                model_dev_scores[model_name][score_name] = value
            unlabelled_preds = model_util.predict(model, model_params, temp_file_path)
            model_unlabelled_predictions.append(unlabelled_preds)
        for model_name, dev_scores in model_dev_scores.items():
            print(f'Model {model_name}: {dev_scores}')
        
        model_pair_pred_overlap = []
        for model_index_1, model_name_1 in enumerate(model_names):
            for model_index_2, model_name_2 in enumerate(model_names):
                if model_name_1 == model_name_2:
                    continue
                model_pred_1 = model_unlabelled_predictions[model_index_1]
                model_pred_2 = model_unlabelled_predictions[model_index_2]
                num_pred_overlap = prediction_overlap(model_pred_1, model_pred_2)
                model_pair_pred_overlap.append((model_name_1, model_name_2, num_pred_overlap))
        print(f'Number of samples in unlabelled data: {len(model_unlabelled_predictions[0])}')
        print('Prediction overlaps:')
        for name_1, name_2, overlap in model_pair_pred_overlap:
            print(f'{name_1} {name_2} {overlap}')
        all_three_overlap = prediction_overlap(*model_unlabelled_predictions)
        print(f'Overlap between all three predictors: {all_three_overlap}')