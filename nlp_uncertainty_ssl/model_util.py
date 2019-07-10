import random
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any, Optional, Tuple

from allennlp.common.params import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models import Model
from allennlp.models.archival import load_archive, archive_model, CONFIG_NAME, Archive
from allennlp.predictors import Predictor
from allennlp.training.trainer import Trainer
import numpy as np
import torch

import nlp_uncertainty_ssl
from nlp_uncertainty_ssl.dataset_readers.emotion import EmotionDatasetReader
from nlp_uncertainty_ssl.models.emotion_classifier import EmotionClassifier
from nlp_uncertainty_ssl.emotion_metrics import f1_metric, jaccard_index

def set_random_env() -> None:
    '''
    `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#L178-L207>`_
    '''
    random.seed(random.choice(range(1, 100000)))
    np.random.seed(random.choice(range(1, 100000)))
    torch.manual_seed(random.choice(range(1, 100000)))
    torch.cuda.manual_seed_all(random.choice(range(1, 100000)))

def read_dataset(dataset_fp: Path, incl_labels: bool, 
                 vocab: Vocabulary) -> List[Dict[str, Any]]:
    '''
    :param dataset_fp: File Path to a list of JSON formatted data
    :param incl_labels: Wether to add the extra `label_array` key/value
    :param vocab: Vocab of the model that is going to predict on the data
    :returns: The data from the dataset with optionally the extra `label_array` 
              key that contains the labels in one hot format.
    '''
    samples = []
    token_to_index = vocab.get_token_to_index_vocabulary(namespace='labels')
    num_labels = vocab.get_vocab_size('labels')
    with dataset_fp.open('r') as dataset_file:
        for line in dataset_file:
            sample = json.loads(line)
            if incl_labels:
                labels = sample['labels']
                label_array = [0] * num_labels
                for label in labels:
                    label_index = token_to_index[label]
                    label_array[label_index] = 1
                sample['label_array'] = label_array
            samples.append(sample)
    return samples

def to_label_arrays(list_dicts: List[Dict[str, Any]], 
                    key: str) -> np.ndarray:
    '''
    :param list_dicts: A list of dictionaries that contain a key which 
                       is a list of labels in one hot format
    :param key: Key whose value in the dictionary is a one hot formatted 
                list of length `k` where `k` is the number of labels
    :returns: All of the one hot encoded formatted lists as a numpy array of 
              shape (num_samples (N), num_labels (k))
    '''
    label_array = []
    for _dict in list_dicts:
        label_array.append(_dict[key])
    return np.array(label_array)

def get_predictions(data_fp: Path, predictor: Predictor,
                    incl_labels: bool, vocab: Vocabulary) -> List[Dict[str, Any]]:
    '''
    :param data_fp: File Path to the dataset file that you wish to predict on
    :param predictor: A predictor that can be used to generate predictions
    :param incl_labels: Wether or not to include the original gold labels in 
                        the return
    :param vocab: Required to get the original gold labels
    :returns: A List of dictionaries that store the data read from the dataset 
              file and with included predictions and optional gold labels
    '''
    data_samples = read_dataset(data_fp, incl_labels=incl_labels, vocab=vocab)
    data_samples = iter(data_samples)
    batch_size = 64
    data_exists = True
    new_data_samples = []
    while data_exists:
        data_batch = []
        for _ in range(batch_size):
            try:
                data_batch.append(next(data_samples))
            except StopIteration:
                data_exists = False
        if data_batch:
            predictions = predictor.predict_batch_json(data_batch)
            for prediction, data_sample in zip(predictions, data_batch):
                data_sample['prediction'] = prediction['labels']
                new_data_samples.append(data_sample)
    return new_data_samples       

def train_model(train_fp: Path, dev_fp: Path, model_fp: Path,
                vocab_data_fps: Optional[List[Path]] = None) -> Tuple[Model, Params]:
    '''
    :param train_fp: The Traning dataset file path
    :param dev_fp: The development dataset file path
    :param model_fp: The json file that describes the model
    :param vocab_data_fps: An optional List of additional dataset files that 
                           will be used to create the models vocab
    :returns: A tuple containing the Trained model and an object that 
              describes the model.
    '''
    set_random_env()
    model_params = Params.from_file(model_fp)
    emotion_dataset_reader = DatasetReader.from_params(model_params.pop('dataset_reader'))

    # Data
    train_dataset = emotion_dataset_reader.read(cached_path(str(train_fp)))
    dev_dataset = emotion_dataset_reader.read(cached_path(str(dev_fp)))
    vocab_datasets = [train_dataset, dev_dataset]
    if vocab_data_fps:
        for vocab_data_fp in vocab_data_fps:
            vocab_datasets.append(emotion_dataset_reader.read(cached_path(str(vocab_data_fp))))
    vocab_data = []
    for vocab_dataset in vocab_datasets:
        vocab_data.extend(vocab_dataset)
    vocab = Vocabulary.from_instances(vocab_data)
    emotion_model = Model.from_params(vocab=vocab, params=model_params.pop('model'))
    data_iter = DataIterator.from_params(model_params.pop('iterator'))
    data_iter.index_with(vocab)
    # Trainer
    with tempfile.TemporaryDirectory() as serial_dir:
        trainer_params = model_params.pop('trainer')
        trainer = Trainer.from_params(model=emotion_model, serialization_dir=serial_dir,
                                    iterator=data_iter, train_data=train_dataset, 
                                    validation_data=dev_dataset, params=trainer_params)
        _ = trainer.train()

        temp_config_fp = str(Path(serial_dir, CONFIG_NAME).resolve())
        Params.from_file(model_fp).to_file(temp_config_fp)
        vocab.save_to_files(Path(serial_dir, "vocabulary").resolve())
        archive_model(serial_dir, files_to_archive=model_params.files_to_archive)
        model_archive = load_archive(serial_dir, cuda_device=0)
        return model_archive.model, model_archive.config

def predict(model: Model, model_params: Params, data_fp: Path
            ) -> List[Dict[str, Any]]:
    '''
    :param model: Model to be used to generate the predictions
    :param params: An object that describes the model
    :param data_fp: File path to be used to predict on
    :returns: The data from the data file path with predictions.
    '''
    model.eval()
    archive = Archive(model=model, config=model_params)
    predictor = Predictor.from_archive(archive, 'emotion-classifier')
    predicted_samples = get_predictions(data_fp, predictor, incl_labels=False, 
                                        vocab=model.vocab)
    return predicted_samples

def dataset_scores(data_fp: Path, model: Model, model_params: Params) -> Dict[str, float]:
    data_predictions = predict(model, model_params, data_fp)
    data_pred_labels = to_label_arrays(data_predictions, 'prediction')
    data_gold = read_dataset(data_fp, True, model.vocab)
    data_gold_labels = to_label_arrays(data_gold, 'label_array')

    scores = {}
    scores['jaccard_index'] = jaccard_index(data_pred_labels, data_gold_labels, 
                                             incl_neutral=True)
    scores['Macro_F1'] = f1_metric(data_pred_labels, data_gold_labels, 
                                   macro=True, incl_neutral=True)
    scores['Micro_F1'] = f1_metric(data_pred_labels, data_gold_labels, 
                                   macro=False, incl_neutral=True)
    return scores