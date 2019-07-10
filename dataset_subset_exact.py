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
                        help='Dataset to remove samples from')
    parser.add_argument('new_dataset_fp', type=parse_path, 
                        help='File to save the new sub sampled dataset')
    parser.add_argument('num_samples_to_keep', type=int,
                        help='Number of samples from the dataset to keep')
    parser.add_argument('--with_replacement', action='store_true')
    args = parser.parse_args()

    if args.with_replacement:
        samples = []
        with args.dataset_subset_fp.open('r') as dataset_file:
            for line in dataset_file:
                samples.append(json.loads(line))
        sub_samples = random.choices(samples, k=args.num_samples_to_keep)
        with args.new_dataset_fp.open('w+') as subset_dataset_file:
            for index, sample in enumerate(sub_samples):
                sample = json.dumps(sample)
                if index != 0:
                    subset_dataset_file.write(f'\n{sample}')
                else:
                    subset_dataset_file.write(f'{sample}')
    else:
        print(args.num_samples_to_keep)
        dataset_fp = args.dataset_subset_fp
        subset_dataset_fp = args.new_dataset_fp
        number_samples = 0
        with dataset_fp.open('r') as dataset_file:
            for line in dataset_file:
                number_samples += 1
        print(f'Number of samples originally: {number_samples}')
        index_to_keep = random.sample(range(number_samples), k=args.num_samples_to_keep)
        first_line = True
        with subset_dataset_fp.open('w+') as subset_dataset_file:
            with dataset_fp.open('r') as dataset_file:
                for index, line in enumerate(dataset_file):
                    if index in index_to_keep:
                        if first_line:
                            first_line = False
                            subset_dataset_file.write(f'{line.strip()}')
                        else:
                            subset_dataset_file.write(f'\n{line.strip()}')
