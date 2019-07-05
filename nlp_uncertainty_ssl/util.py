import copy
from typing import List
import re

import twokenize

def is_character_preserving(original_text: str, text_tokens: List[str]
                            ) -> bool:
    '''
    :param original_text: Text that has been tokenized
    :param text_tokens: List of tokens after the text has been tokenized
    :returns: True if the tokenized text when all characters are joined 
                together is equal to the original text with all it's 
                characters joined together.
    '''
    text_tokens_copy = copy.deepcopy(text_tokens)
    # Required as some of the tokenization tokens contain whitespace at the 
    # end of them I think this due to Stanford method being a Neural Network
    text_tokens_copy = [token.strip(' ') for token in text_tokens_copy]
    tokens_text = ''.join(text_tokens_copy)
    original_text = ''.join(original_text.split())
    if tokens_text == original_text:
        return True
    else:
        return False

def tweet_tokenizer(text: str) -> List[str]:
    '''
    A Twitter tokenizer from
    `CMU Ark <https://github.com/brendano/ark-tweet-nlp>`_
    This is a wrapper of
    `this <https://github.com/Sentimentron/ark-twokenize-py>`_. Further more 
    this is an adapted version as it also splits hashtags 
    e.g. `#anything` becomes [`#`, `anything`]. This follows the tokenization 
    of `Yu et al. 2018 <https://www.aclweb.org/anthology/D18-1137>`_. 

    :param text: A string to be tokenized.
    :returns: A list of tokens where each token is a String.
    :raises AssertionError: If the tokenized text is not character preserving.
    :raises ValueError: If the given text is not a String
    '''

    hashtag_pattern = re.compile('^#.+')
    if isinstance(text, str):
        tokenized_text = twokenize.tokenizeRawTweetText(text)
        print(tokenized_text)
        hashtag_tokenized_text = []
        for token in tokenized_text:
            if hashtag_pattern.search(token):
                hashtag = token[0]
                hashtag_tokenized_text.append(hashtag)
                
                other_token_text = token[1:].strip()
                if other_token_text:
                    hashtag_tokenized_text.append(other_token_text)
            else:
                hashtag_tokenized_text.append(token)

        assert_err = ('The tokenizer has not been charcter preserving. Original'
                      f' text: {text}\nHashtag tokenized tokens '
                      f'{hashtag_tokenized_text}')
        assert is_character_preserving(text, hashtag_tokenized_text), assert_err

        return hashtag_tokenized_text

    raise ValueError(f'The paramter must be of type str not {type(text)}')