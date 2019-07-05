from typing import List

import pytest

from nlp_uncertainty_ssl.util import tweet_tokenizer, is_character_preserving

def test_tweet_tokenizer():
    non_hashtag = ' This is some text '
    hashtag_text = 'something with multiple# hashtag #etc#lol #eg'
    nothing = '   '
    examples = [non_hashtag, hashtag_text, nothing]
    
    non_hashtag_answer = ['This', 'is', 'some', 'text']
    hashtag_text_answer = ['something', 'with', 'multiple#', 'hashtag',
                           '#', 'etc', '#', 'lol', '#', 'eg']
    nothing_answer = []
    answers = [non_hashtag_answer, hashtag_text_answer, nothing_answer]

    for answer, example in zip(answers, examples):
        example = tweet_tokenizer(example)
        assert answer == example
    with pytest.raises(ValueError):
        tweet_tokenizer(1)

def not_char_preserving_tokenizer(text: str) -> List[str]:
    tokens = text.split()
    alt_tokens = []
    for token in tokens:
        if token == "other's":
            alt_tokens.append('other')
        else:
            alt_tokens.append(token)
    return alt_tokens

def test_is_character_preserving():
    sentence = "This is a other's sentence to test"
    tokens = str.split(sentence)
    assert is_character_preserving(sentence, tokens)

    tokens = not_char_preserving_tokenizer(sentence)
    assert not is_character_preserving(sentence, tokens)
