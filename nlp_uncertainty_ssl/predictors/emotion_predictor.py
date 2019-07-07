from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.predictors.predictor import Predictor

@Predictor.register('emotion-classifier')
class EmotionPredictor(Predictor):
    """
    Predictor for the 
    :class:`nlp_uncertainty_ssl.models.emotion_classifier.EmotionClassifier` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like either, the JSON can have the option 
        of having "ID" key:
        
        1. ``{"text": "...", "tokens": ["..."]}``
        """
        input_dict = {}
        input_dict['tokens'] = [Token(token) for token in json_dict['tokens']]
        input_dict['text'] = json_dict['text']
        if 'ID' in json_dict:
            input_dict['ID'] = json_dict['ID']
        return self._dataset_reader.text_to_instance(**input_dict)