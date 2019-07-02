from pathlib import Path

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
            
    def test_label_encoding_required(self):
        # calculate_span_f1 requires label encoding
        params = Params.from_file(self.config_file).duplicate()
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