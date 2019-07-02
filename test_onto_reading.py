from pathlib import Path
from typing import Iterable

from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence, to_bioul


def _normalize_word(word: str):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def _ontonotes_subset(ontonotes_reader: Ontonotes,
                      file_path: str,
                      domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if (domain_identifier is None or f"/{domain_identifier}/" in conll_file) and "/pt/" not in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)
# This can be either a certain data split e.g. test or the larger directory of `data`
onto_notes_path = str(Path('..', 'conll-formatted-ontonotes-5.0-12', 
                           'conll-formatted-ontonotes-5.0', 'conll-2012', 
                           'v4', 'data', 'test').resolve())

ontonotes_reader = Ontonotes()
for sentence in _ontonotes_subset(ontonotes_reader, onto_notes_path, None):
    tokens = [Token(_normalize_word(t)) for t in sentence.words]
    ne = sentence.named_entities
    print('yes')