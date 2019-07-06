import logging
import json
from typing import Dict, Any, Optional, List

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, MultiLabelField, MetadataField, Field
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("emotion")
class EmotionDatasetReader(DatasetReader):
    '''
    Dataset reader designed to read a list of JSON like objects of the 
    following type:
    {"ID": "2017-En-21441", 
     "text": "\u201cWorry is a down payment on a problem you may never have'. 
              \u00a0Joyce Meyer.  #motivation #leadership #worry", 
     "tokens": ["\u201c", "Worry", "is", "a", "down", "payment", "on", "a", 
                "problem", "you", "may", "never", "have", "'", ".", "Joyce", 
                "Meyer", ".", "#", "motivation", "#", "leadership", "#", 
                "worry"], 
     "labels": ["anticipation", "optimism", "trust"]}
     
    :returns: A ``Dataset`` of ``Instances`` for Target Extraction.
    '''
    def __init__(self, lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as emotion_file:
            logger.info("Reading Emotion instances from jsonl dataset at: %s", 
                        file_path)
            for line in emotion_file:
                example = json.loads(line)
                example_instance: Dict[str, Any] = {}

                multiple_emotion_labels = example["labels"]
                tokens_ = example["tokens"]
                # TextField requires ``Token`` objects
                tokens = [Token(token) for token in tokens_]

                example_instance['labels'] = multiple_emotion_labels
                example_instance['tokens'] = tokens
                example_instance['text'] = example["text"]
                example_instance['ID'] = example['ID']
                yield self.text_to_instance(**example_instance)
    
    def text_to_instance(self, tokens: List[Token],
                         text: str, ID: Optional[str] = None, 
                         labels: Optional[List[str]] = None) -> Instance:
        '''
        The tokens are expected to be pre-tokenised.

        :param tokens: The text that has been tokenised
        :param text: The text from the sample
        :param ID: The ID of the sample
        :param labels: A list of labels (can be an empty list which is 
                       associated implictly to the neutral class)
        :returns: An Instance object with all of the above enocded for a
                  PyTorch model.
        '''
        token_sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': token_sequence}

        meta_fields = {}
        meta_fields["words"] = [x.text for x in tokens]
        meta_fields["text"] = text
        if ID is not None:
            meta_fields["ID"] = ID
        instance_fields["metadata"] = MetadataField(meta_fields)

        if labels is not None:
            instance_fields['labels'] = MultiLabelField(labels, 
                                                        label_namespace="labels")

        return Instance(instance_fields)