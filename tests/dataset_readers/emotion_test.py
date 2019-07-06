from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from nlp_uncertainty_ssl.dataset_readers.emotion import EmotionDatasetReader

class TestEmotionDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool):
        reader = EmotionDatasetReader(lazy=lazy)

        data = Path(__file__, '..', 'data', 'emotion_dataset.json').resolve()

        instance1 = {"tokens": ["#", "smile", "every", "morning", "to", "a", 
                                "positive", "head", "start", "with", "your", 
                                "#", "clients", "relations"],
                     "labels": ["joy", "optimism"],
                     "text": "#smile every morning to a positive head "\
                             "start with your #clients relations",
                     "ID": "2017-En-30296"}

        instance2 = {"tokens": ["I", "saved", "him", "after", "ordering", 
                                "him", "to", "risk", "his", "life", ".", "I", 
                                "didn't", "panic", "but", "stayed", "calm", 
                                "and", "rescued", "him", "."],
                     "text": "I saved him after ordering him to risk his life."\
                             " I didn't panic but stayed calm and rescued him.",
                     "ID": "2017-En-20473",
                     "labels": ["optimism"]}

        instance3 = {"ID": "2017-En-10565",
                     "text": "@SlaveGuinevere its more of a little prick "\
                             "than a sting .......... but you have to come to"\
                             " Tennessee to get it1",
                     "tokens": ["@SlaveGuinevere", "its", "more", "of", "a", 
                                "little", "prick", "than", "a", "sting", 
                                "..........", "but", "you", "have", "to", 
                                "come", "to", "Tennessee", "to", "get", "it1"],
                     "labels": []}

        instances = ensure_list(reader.read(str(data)))
        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"]] == instance1["tokens"]
        assert fields['labels'].labels == instance1["labels"]
        assert fields['metadata']['words'] == instance1["tokens"]
        assert fields['metadata']['ID'] == instance1["ID"]
        assert fields['metadata']['text'] == instance1["text"]
        
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"]] == instance2["tokens"]
        assert fields['labels'].labels == instance2["labels"]
        assert fields['metadata']['words'] == instance2["tokens"]
        assert fields['metadata']['ID'] == instance2["ID"]
        assert fields['metadata']['text'] == instance2["text"]

        fields = instances[2].fields
        assert [t.text for t in fields["tokens"]] == instance3["tokens"]
        assert fields['labels'].labels == instance3["labels"]
        assert fields['metadata']['words'] == instance3["tokens"]
        assert fields['metadata']['ID'] == instance3["ID"]
        assert fields['metadata']['text'] == instance3["text"]