import argparse
import json
from pathlib import Path
import random

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_subset_1_fp', type=parse_path, 
                        help='Dataset 1')
    parser.add_argument('dataset_subset_2_fp', type=parse_path, 
                        help='Dataset 2')
    parser.add_argument('new_dataset_fp', type=parse_path, 
                        help='File to save the new sub sampled dataset')
    parser.add_argument('num_dataset_1', type=int,
                        help='Number of samples from dataset 1')
    parser.add_argument('num_dataset_2', type=int,
                        help='Number of samples from dataset 2')
    args = parser.parse_args()

    all_samples = []
    samples_1 = []
    with args.dataset_subset_1_fp.open('r') as dataset_file:
        for line in dataset_file:
            samples_1.append(json.loads(line))
    to_keep = random.sample(range(len(samples_1)), k=args.num_dataset_1)
    for index, sample in enumerate(samples_1):
        if index in to_keep:
            all_samples.append(sample)
    print(len(all_samples))
    samples_2 = []
    with args.dataset_subset_2_fp.open('r') as dataset_file:
        for line in dataset_file:
            samples_2.append(json.loads(line))
    to_keep = random.sample(range(len(samples_2)), k=args.num_dataset_2)
    for index, sample in enumerate(samples_2):
        if index in to_keep:
            all_samples.append(sample)
    print(len(all_samples))
    with args.new_dataset_fp.open('w+') as subset_dataset_file:
        for index, sample in enumerate(all_samples):
            sample = json.dumps(sample)
            if index != 0:
                subset_dataset_file.write(f'\n{sample}')
            else:
                subset_dataset_file.write(f'{sample}')