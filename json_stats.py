import argparse
import json
from pathlib import Path

from nlp_uncertainty_ssl.util import simple_stats

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    json_dataset_fps_help = 'File path(s) to the json dataset to get '\
                            'statistics from'
    normalise_by_sample_count_help = 'Whether to normalise the statistics by '\
                                     'the number of sample'

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dataset_fps", type=parse_path, nargs='+',
                        help=json_dataset_fps_help)
    parser.add_argument("--normalise_by_sample_count", action='store_true',
                        help=normalise_by_sample_count_help)
    args = parser.parse_args()

    json_fps = args.json_dataset_fps
    json_data = []
    for json_fp in json_fps:
        with json_fp.open('r') as json_file:
            json_data.extend(json.load(json_file))
    number_samples = len(json_data)
    data_stats = simple_stats(json_data)
    if args.normalise_by_sample_count:
        data_stats = {name: f'{(stat/number_samples) * 100:.1f}'
                      for name, stat in data_stats.items()}
    data_stats = dict(sorted(data_stats.items(), key=lambda x: x[0]))
    print(data_stats)
    import pdb
    pdb.set_trace()