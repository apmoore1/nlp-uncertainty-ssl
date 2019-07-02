from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

@Predictor.register('cvt-tagger')
class CvtTaggerPredictor(Predictor):
    """
    Predictor for the 
    :class:`nlp_uncertainty_ssl.models.cvt_tagger.CvtTagger` model.
    This predictor is very much based on the 
    :class:`from allennlp.predictors.sentence.SentenceTaggerPredictor`
    The main difference:
    
    1. The option to use either the tokenizer that is in the constructor of the 
       class or to provide the tokens within the JSON that is to be processed 
       thus allowing the flexiability of using your own custom tokenizer.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, 
                 language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like either:
        
        1. ``{"sentence": "..."}``
        2. ``{"sentence": "...", "tokens": ["..."]}``
        3. ``{"tokens": ["..."]}``

        The first case will tokenize the text using the tokenizer in the 
        constructor. The later two will just use the tokens given.

        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        if 'tokens' in json_dict:
            tokens = [Token(token) for token in json_dict['tokens']]
        else:
            sentence = json_dict["sentence"]
            tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)