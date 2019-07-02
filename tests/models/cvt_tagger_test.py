from pathlib import Path
import tempfile

from allennlp.common.testing import ModelTestCase
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
import pytest

import nlp_uncertainty_ssl

class CvtTaggerTest(ModelTestCase):
    '''
    Using the same data as those used in the ``crf_tagger`` test from allennlp
    '''

    def setUp(self):
        data_fp = Path(__file__, '..', 'data', 'conll2003.txt').resolve()
        self.data_file = str(data_fp)

        config_dir = Path(__file__, '..', 'configs').resolve()
        config_fp = Path(config_dir, 'cvt_tagger.json')
        self.config_file = str(config_fp)

        self.set_up_model(self.config_file, self.data_file)
        super().setUp()

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.config_file)

    def test_tagger_with_dropout_save_and_load(self):
        params = Params.from_file(self.config_file).duplicate()
        params['model']['dropout'] = 0.5
        with tempfile.NamedTemporaryFile() as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
    
    def test_tagger_without_feedforward_save_load(self):
        params = Params.from_file(self.config_file).duplicate()
        values = ['feedforward']
        for value in values:
            params['model'].pop(value)
        with tempfile.NamedTemporaryFile() as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_tagger_without_feedforward_encoder_save_load(self):
        params = Params.from_file(self.config_file).duplicate()
        values = ['feedforward', 'encoder']
        for value in values:
            params['model'].pop(value)
        with tempfile.NamedTemporaryFile() as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
            
    def test_label_encoding_required(self):
        # calculate_span_f1 requires label encoding
        params = Params.from_file(self.config_file).duplicate()
        params["model"]["calculate_span_f1"] = True
        params["model"].pop("label_encoding")
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.config_file).duplicate()
        # Make the encoder wrong - it should be 210 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 200
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        # Make the encoder output wrong should be 300 not 70
        params = Params.from_file(self.param_file).duplicate()
        params["model"]["encoder"]["hidden_size"] = 70
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        # Remove the encoder and force an error with the text field embedder
        params = Params.from_file(self.param_file).duplicate()
        params["model"].pop("encoder")
        # Text embedder is output 210 and feed forward expects 600
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}
        
        assert 6 == len(output_dict)
        output_keys = ['logits', 'mask', 'tags', 'class_probabilities',
                       'loss', 'words']
        for key in output_keys:
            assert key in output_dict