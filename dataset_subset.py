import argparse
import json
from pathlib import Path
import random

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_subset_fp', type=parse_path, 
                        help='Dataset to remove labels from')
    parser.add_argument('new_dataset_fp', type=parse_path, 
                        help='File to save the new sub sampled dataset')
    parser.add_argument('removed_dataset_fp', type=parse_path,
                        help='File path to the data that has been removed')
    parser.add_argument('percentage_to_remove', type=int,
                        help='Percentage of the dataset to sub-sample')
    args = parser.parse_args()
    
    print(args.percentage_to_remove)
    prob_to_remove = args.percentage_to_remove / 100.0
    prob_to_keep = 1 - prob_to_remove
    print(f'Prob to keep: {prob_to_keep}\nProb to remove: {prob_to_remove}')
    dataset_fp = args.dataset_subset_fp
    subset_dataset_fp = args.new_dataset_fp
    first_line = True
    other_first_line = True
    with args.removed_dataset_fp.open('w+') as removed_dataset_file:
        with subset_dataset_fp.open('w+') as subset_dataset_file:
            with dataset_fp.open('r') as dataset_file:
                for line in dataset_file:
                    to_keep = random.choices([0, 1], weights=[prob_to_remove, prob_to_keep])[0]
                    if to_keep:
                        if first_line:
                            first_line = False
                            subset_dataset_file.write(f'{line.strip()}')
                        else:
                            subset_dataset_file.write(f'\n{line.strip()}')
                    else:
                        if other_first_line:
                            other_first_line = False
                            removed_dataset_file.write(f'{line.strip()}')
                        else:
                            removed_dataset_file.write(f'\n{line.strip()}')
