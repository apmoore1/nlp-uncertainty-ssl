import argparse
from collections import defaultdict
import json
import statistics
from pathlib import Path

from nlp_uncertainty_ssl import model_util

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", type=parse_path,
                        help="File path to the training dataset")
    parser.add_argument("dev_fp", type=parse_path,
                        help="File path to the development dataset")
    parser.add_argument("test_fp", type=parse_path,
                        help='File path to the test dataset')
    parser.add_argument("model_fp", type=parse_path,
                        help='File path to the model config file')
    parser.add_argument("save_dir", type=parse_path,
                        help='File path to the save directory to stroe the results')
    args = parser.parse_args()
    train_fp = args.train_fp
    dev_fp = args.dev_fp
    test_fp = args.test_fp

    model_fp = args.model_fp

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_dev_results_fp = Path(args.save_dir, 'dev.json').resolve()
    save_test_results_fp = Path(args.save_dir, 'test.json').resolve()

    dev_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for i in range(5):
        model, model_params = model_util.train_model(train_fp, dev_fp, model_fp, vocab_data_fps=[test_fp])
        for score_name, value in model_util.dataset_scores(dev_fp, model, model_params).items():
            dev_scores[score_name].append(value)
        for score_name, value in model_util.dataset_scores(test_fp, model, model_params).items():
            test_scores[score_name].append(value)
        print(dev_scores)
        print(test_scores)

    with save_dev_results_fp.open('w+') as dev_results_file:
        json.dump(dev_scores, dev_results_file)
    with save_test_results_fp.open('w+') as test_results_file:
        json.dump(test_scores, test_results_file)
    print(f'Development averages:')
    for score_name, scores in dev_scores.items():
        print(f'{score_name}: {statistics.mean(scores)}')
    print(f'\nTest averages:')
    for score_name, scores in test_scores.items():
        print(f'{score_name}: {statistics.mean(scores)}')