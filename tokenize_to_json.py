import argparse
from collections import Counter
import html
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from nlp_uncertainty_ssl.util import tweet_tokenizer, simple_stats

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def process_tsv_file(data_path: Path) -> List[Dict[str, Any]]:
    '''
    :param data_path: File Path to one of the train, dev, test datasets from 
                      the emotion datasets of `Mohammad et al. 2018 
                      <http://saifmohammad.com/WebDocs/semeval2018-task1.pdf>`_
    :returns: That datasets as a List of dictionaries where the dictionaries 
              contain the following keys and values:

              1. text: Text of the Tweet (str)
              2. tokens: Text tokenized using the same method as `Yu et al. 
                 2018 <https://www.aclweb.org/anthology/D18-1137>`_ (List[str])
              3. ID: ID that is associated to the sample (str)
              4. labels: A list of all of the emotion labels associated to this 
                 sample, the list can be empty which indicates neutral 
                 (List[str])
    '''
    data = pd.read_csv(data_path, sep='\t')
    index_to_column = {}
    for index, column in enumerate(data.columns):
        index_to_column[index] = column.strip()
    index_to_column_length = len(index_to_column)

    json_samples: List[Dict[str, Any]] = []
    for _, sample in data.iterrows():
        sample = sample.tolist()
        sample_length = len(sample)
        assert sample_length == index_to_column_length

        json_sample = {}
        labels = []
        for index in range(len(sample)):
            column_name = index_to_column[index]
            if column_name == 'ID':
                json_sample['ID'] = sample[index]
            elif column_name == 'Tweet':
                text = sample[index]
                text = html.unescape(text)
                # This is required as some of the escaped symbols are next to 
                # each other and therefor the unescape has to be performed 
                # twice
                if '&ldquo;' in text or '&rdquo;' in text or '&amp;' in text:
                    text = html.unescape(text)
                json_sample['text'] = text
                json_sample['tokens'] = tweet_tokenizer(text)
            else:
                if sample[index]:
                    labels.append(column_name)
        json_sample['labels'] = labels
        json_samples.append(json_sample)
    return json_samples

if __name__ == '__main__':
    dataset_stats_fp_help = 'File path to store the dataset stats in a Latex table format'

    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", type=parse_path,
                        help="File path to the training dataset")
    parser.add_argument("dev_fp", type=parse_path,
                        help='File path to the development dataset')
    parser.add_argument("test_fp", type=parse_path,
                        help='File path to the test dataset')
    parser.add_argument("dataset_stats_fp", type=parse_path, 
                        help=dataset_stats_fp_help)
    args = parser.parse_args()

    train_path = args.train_fp
    dev_path = args.dev_fp
    test_path = args.test_fp
    data_fps = [train_path, dev_path, test_path]
    names = ['train', 'development', 'test']

    overall_data_stats = {}
    total_number_samples = 0
    for name, data_path in zip(names, data_fps):
        json_data = process_tsv_file(data_path)
        # Write json data to file
        data_path: Path
        json_data_fp = data_path.with_name(f'{name}.json')
        with json_data_fp.open('w+') as json_data_file:
            for index, sample in enumerate(json_data):
                sample = json.dumps(sample)
                if index != 0:
                    sample = f'\n{sample}'
                json_data_file.write(sample)
        # Get general data stats from the data
        stats = sorted(simple_stats(json_data).items(), key=lambda x: x[0])
        overall_data_stats[name] = dict(stats)
        total_number_samples += len(json_data)

    # Columns are ['Train', 'Dev', 'Test'], indexs are label names
    data_stats_df = pd.DataFrame(overall_data_stats)
    # This sums the number of lables across the data splits
    data_stats_df = data_stats_df.sum(axis=1)
    # Divide the number of labels by the number of samples and convert to %
    data_stats_df = (data_stats_df / total_number_samples) * 100
    # Round
    data_stats_df = data_stats_df.round(1)
    # Write to latex
    data_stats_df = pd.DataFrame(dict(data_stats_df),index=[0])
    dataset_stats_fp: Path = args.dataset_stats_fp
    dataset_stats_fp.parent.mkdir(parents=True, exist_ok=True)
    with dataset_stats_fp.open('w+') as dataset_stats_file:
        data_stats_df.to_latex(dataset_stats_file)
    print(f'Data statistics breakdown:\n{data_stats_df}')