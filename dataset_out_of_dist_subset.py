import argparse
import json
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fp', type=parse_path, 
                        help='Dataset to get out of distribution label samples from')
    parser.add_argument('new_dataset_fp', type=parse_path, 
                        help='File to save the new datasets that only contains samples with out of distribution labels')
    parser.add_argument("out_of_distribution_labels", type=str, nargs='+',
                        help='The labels that are out of distribution')
    args = parser.parse_args()
    
    print(args.out_of_distribution_labels)

    out_of_distribution_labels = args.out_of_distribution_labels
    appears = {}
    first_line = True
    with args.new_dataset_fp.open('w+') as new_dataset_file:
        with args.dataset_fp.open('r') as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                skip_sample = False
                # Do not include neutral samples
                if sample['labels'] == []:
                    continue
                # Ensure that no other label is in the sample
                for label in sample['labels']:
                    if label not in out_of_distribution_labels:
                        skip_sample = True
                if skip_sample:
                    continue
                # Ensure that the only labels in the sample are those from 
                # the out_of_distribution_labels
                skip_sample = True
                for label in sample['labels']:
                    if label in out_of_distribution_labels:
                        skip_sample = False
                        appears[label] = True
                if skip_sample:
                    continue
                if first_line:
                    first_line = False
                    new_dataset_file.write(f'{line.strip()}')
                else:
                    new_dataset_file.write(f'\n{line.strip()}')
    print(appears)
