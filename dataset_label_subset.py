import argparse
import json
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_subset_fp', type=parse_path, 
                        help='Dataset to remove labels from')
    parser.add_argument('new_dataset_fp', type=parse_path, 
                        help='File to save the new sub sampled dataset')
    parser.add_argument("labels_to_remove", type=str, nargs='+',
                        help='The labels to remove from the dataset')
    args = parser.parse_args()
    
    print(args.labels_to_remove)
    disinclude_labels = args.labels_to_remove
    appears = {}
    dataset_fp = args.dataset_subset_fp
    subset_dataset_fp = args.new_dataset_fp
    first_line = True
    with subset_dataset_fp.open('w+') as subset_dataset_file:
        with dataset_fp.open('r') as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                skip_sample = False
                for label in sample['labels']:
                    if label in disinclude_labels:
                        appears[label] = True
                        skip_sample = True
                if skip_sample:
                    continue
                if first_line:
                    first_line = False
                    subset_dataset_file.write(f'{line.strip()}')
                else:
                    subset_dataset_file.write(f'\n{line.strip()}')
    print(appears)
