from pathlib import Path
from typing import Dict, List, Any

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import pytest

import nlp_uncertainty_ssl

class TestEmotionPredictor():
    @pytest.mark.parametrize("incl_id", (True, False))
    def test_standard_use(self, incl_id: bool):
        def keys_in_output(keys: List[str], output: Dict[str, Any]) -> None:
            for key in keys:
                assert key in output

        token_input = ["#", "smile", "every", "morning", "to", 
                       "a", "positive", "head", "start", "with", 
                       "your", "#", "clients", "relations"]
        text_input = "#smile every morning to a positive head "\
                     "start with your #clients relations"
        example_input = {"text": text_input, "tokens": token_input}
        

        archive_dir = Path(__file__, '..', 'saved_models').resolve()
        archive_model = load_archive(str(Path(archive_dir, 'emotion_classifier', 'model.tar.gz')))
        predictor = Predictor.from_archive(archive_model, 'emotion-classifier')
        
        output_keys = ['logits', 'probs', 'labels', 'words', 'text', 'readable_labels']
        if incl_id:
            example_input['ID'] = '123'
            output_keys.append('ID')

        result = predictor.predict_json(example_input)
        keys_in_output(output_keys, result)
        assert result['words'] == token_input
        assert result['text'] == text_input
        if incl_id:
            assert result['ID'] == '123'
        assert len(output_keys) == len(result)