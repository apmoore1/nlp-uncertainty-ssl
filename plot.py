from typing import List
from pathlib import Path
import json

import pandas as pd
import seaborn as sns

def test_results(save_dir: Path) -> List[float]:
    with Path(save_dir, 'test.json').open('r') as test_file:
        return json.load(test_file)['jaccard_index']
top_dir = Path('.', 'results', 'emotion', 'ssl')
labels_percent = []
scores = []
file_paths_percent = [(Path(top_dir, 'ood'), 100), (Path(top_dir, 'normal'), 0),
                      (Path(top_dir, 'ood_25'), 25), (Path(top_dir, 'ood_50'), 50),
                      (Path(top_dir, 'ood_75'), 75)]
for file_path, percent in file_paths_percent:
    for result in test_results(file_path):
        scores.append(result * 100)
        labels_percent.append(percent)
df = pd.DataFrame({'Jaccard Index (%)': scores, '% Out of class distribution in un-labeled data': labels_percent})
sns_plot = sns.lineplot(x='% Out of class distribution in un-labeled data', y='Jaccard Index (%)', data=df, ci='sd', markers=True, err_style='bars')
fig = sns_plot.get_figure()
fig.savefig("output.png")