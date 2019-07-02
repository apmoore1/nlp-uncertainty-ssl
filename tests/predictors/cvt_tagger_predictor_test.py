from pathlib import Path
from typing import Dict, List, Any

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import nlp_uncertainty_ssl

class TestCvtTaggerPredictor():

    def test_standard_use(self):
        def keys_in_output(keys: List[str], output: Dict[str, Any]) -> None:
            for key in keys:
                assert key in output

        example_input = {'sentence': "The laptop's"}
        example_sentence_token_input = {'tokens': ["The", "laptop's", "case", 
                                                   "was", "great", "and", 
                                                   "cover", "was", "rubbish"],
                                        'sentence': "The laptop's case was great"\
                                                    " and cover was rubbish"}
        example_token_input = {'tokens': ["The", "laptop's", "case", "was", 
                                          "great", "and", "cover", "was", 
                                          "rubbish"]}
            

        archive_dir = Path(__file__, '..', 'saved_models').resolve()
        archive_model = load_archive(str(Path(archive_dir, 'cvt_tagger', 'model.tar.gz')))
        predictor = Predictor.from_archive(archive_model, 'cvt-tagger')
        output_keys = ['logits', 'mask', 'tags', 'class_probabilities']

        result = predictor.predict_json(example_input)
        keys_in_output(output_keys, result)
        assert result['words'] == ["The", "laptop", "'s"]

        result = predictor.predict_json(example_sentence_token_input)
        keys_in_output(output_keys, result)
        assert result['words'] == ["The", "laptop's", "case", "was", "great", 
                                   "and", "cover", "was", "rubbish"]
        
        result = predictor.predict_json(example_token_input)
        keys_in_output(output_keys, result)
        assert result['words'] == ["The", "laptop's", "case", "was", "great", 
                                   "and", "cover", "was", "rubbish"]